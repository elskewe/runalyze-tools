import pickle
import webbrowser
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyperclip
from rich.console import Console
from rich.table import Table
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from perfect_clothing import assumptions, load_data
from perfect_clothing.augment_data import augment_data
from perfect_clothing.prepare_data import prepare_data, save_candidate_outfits


def train():
    try:
        # delete the cache file if it is older than the raw data (jumping to the catch if the file
        # is not found is fine as it won't be possible to load it either)
        if Path(assumptions.DATA_FILENAME).stat().st_mtime < Path(load_data.CACHE_FILE).stat().st_mtime:
            Path(assumptions.DATA_FILENAME).unlink()
        with open(assumptions.DATA_FILENAME, "rb") as f:
            (data_df, encoded_clothing_columns) = pickle.load(f)
    except FileNotFoundError:
        data = load_data.get_data()
        data_df, encoded_clothing_columns = prepare_data(data)
        candidate_outfits = save_candidate_outfits(data_df, encoded_clothing_columns)
        data_df = augment_data(data_df, encoded_clothing_columns, candidate_outfits)

        with open(assumptions.DATA_FILENAME, "wb") as f:
            pickle.dump((data_df, encoded_clothing_columns), f)

    train_core(data_df, encoded_clothing_columns, n_estimators=316, num_leaves=31,
               group_activities=False, model_filename=assumptions.MODEL_FILENAME)
    train_core(data_df, encoded_clothing_columns, n_estimators=100, num_leaves=31,
               group_activities=True, model_filename=assumptions.ALTERNATIVE_MODEL_FILENAME)


def train_core(data: pd.DataFrame, encoded_clothing_columns: list[str], n_estimators=-1, num_leaves=31, group_activities=True,
               model_filename=assumptions.MODEL_FILENAME):
    """Actually train the model.

    n_estimator is passed to LGBMClassifier. If it is -1, it will be optimized beforehand by a
    hyperparameter search. `group_activities` determines whether all augmented activities based on
    the same base activity should be kept in the same fold for CV and train/test split.
    """
    x = data[[*assumptions.INPUT_COLUMNS, *encoded_clothing_columns]]
    y = data["comfort_int"]
    train_idx, test_idx = next(
        StratifiedGroupKFold(shuffle=True).split(x, y, data.index if group_activities else range(0, len(x))))
    x_train, x_test, y_train, y_test = x.iloc[train_idx], x.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    weights = weight_dates(data["date_time"])
    weights_train, weights_test = weights.iloc[train_idx], weights.iloc[test_idx]

    # this is a list instead of a generator to allow pickle and use in both the hyperparameter
    # search and actual training
    cv = list(StratifiedGroupKFold(shuffle=True)
              .split(x_train, y_train, data.index[train_idx] if group_activities else range(0, len(train_idx))))

    base = lgb.LGBMClassifier(n_estimators=n_estimators, num_leaves=num_leaves, reg_alpha=0.01, reg_lambda=0.01,
                              verbose=-1, class_weight="balanced", importance_type="gain",)
    if n_estimators == -1:
        param_grid = {
            "n_estimators": np.logspace(1, 2, 3, dtype=int),
            "num_leaves": np.logspace(1, 2, 3, dtype=int),
            "reg_alpha": np.concatenate(([0], np.logspace(-2, 0, 2))),
            "reg_lambda": np.concatenate(([0], np.logspace(-2, 0, 2))),
        }
        # search for best hyperparameters
        search = GridSearchCV(base, param_grid, cv=cv, scoring="f1_macro", error_score="raise",
                              refit=best_low_complexity, verbose=3)
        search.fit(x_train, y_train, sample_weight=weights_train)
        base = search.best_estimator_
        print(search.best_params_)

    # Calibrate for better probabilities (sigmoid works reliably with moderate data)
    clf = CalibratedClassifierCV(base, cv=cv, method='sigmoid')

    clf.fit(x_train, y_train, sample_weight=weights_train)

    y_pred = clf.predict(x_test)
    labels = list(assumptions.TEMPERATURE_LABEL_MAPPING.keys())
    print(classification_report(y_test, y_pred, target_names=labels, sample_weight=weights_test))
    cm = confusion_matrix(y_test, y_pred, sample_weight=weights_test)
    print(cm.astype("int"))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()

    plot_importances(clf, x.columns.to_list())
    find_problematic_regions(x_test, y_test, y_pred)
    # just calling this to set a debugger breakpoint is not nice, but I don't want to think of a
    # good output right now
    problematic_entries = find_problematic_entries(clf, x_test, y_test, y_pred)

    # save model
    with open(model_filename, "wb") as f:
        pickle.dump(clf, f)
    return None


def best_low_complexity(cv_results: dict[str, Any]):
    """Balance model complexity with cross-validated score.

    The best hyperparameter combination is the one whose mean is within one standard error of the
    mean of the best one with the lowest complexity.

    Args:
        cv_results (dict): the result of a GridSearchCV

    Returns:
        tuple: the hyperparameter combination with the highest test score and the lowest complexity
    """
    best_score_idx = np.argmax(cv_results["mean_test_score"])
    best_score_lower_bound = cv_results["mean_test_score"][best_score_idx]\
        - cv_results["std_test_score"][best_score_idx]

    candidate_idx = np.flatnonzero(cv_results["mean_test_score"] >= best_score_lower_bound)
    # choose the candidate with the lowest product of the hyperparameters (should be a good proxy
    # for the complexity) with the mean score being the tie breaker. I don't like that this is
    # hardcoded, but I can't think of a better option to prevent also multiplying with
    # regularization terms (which might be zero without making the model less complex)
    candidates = [(i, cv_results["params"][i]["n_estimators"] * cv_results["params"][i]["num_leaves"],
                   cv_results["mean_test_score"][i]) for i in candidate_idx]
    # choose the result with the lowest complexity and the highest score
    best_candidate = min(candidates, key=lambda c: (c[1], -c[2]))
    return best_candidate[0]


def plot_importances(clf: CalibratedClassifierCV, feature_names: list[str]):

    importances = np.mean([
        est.estimator.feature_importances_
        for est in clf.calibrated_classifiers_
        ], axis=0)

    # Sort by importance
    idx = np.argsort(importances)
    sorted_importances = importances[idx]
    sorted_features = np.array(feature_names)[idx]

    plt.figure()
    plt.barh(sorted_features, sorted_importances)
    plt.xscale('log')
    plt.xlabel("Feature Importance (log scale)")
    plt.title("LightGBM Feature Importances (averaged over folds)")
    plt.tight_layout()
    plt.show()


def find_problematic_regions(x: pd.DataFrame, y: pd.Series, y_pred: np.ndarray, n_bins=10, top_k=5):
    """Prints the feature regions with the worst performance.

    Args:
        x (pd.DataFrame): the features
        y (pd.Series): the ground truth labels
        y_pred (np.ndarray): the predicted labels
        n_bins (int, optional): the number of bins to use. Defaults to 10.
        top_k (int, optional): the number of worst regions to print. Defaults to 5.
    """
    values = []
    for column in x.columns:
        if len(x[column].unique()) < n_bins:
            bounds = x[column].unique()
            bounds.sort()
        else:
            bounds = np.linspace(x[column].min(), x[column].max(), n_bins + 1)
            bounds[-1] += 1
        for lower_bound, upper_bound in zip(bounds[:-1], bounds[1:]):
            idx = (x[column] >= lower_bound) & (x[column] < upper_bound)
            by_label = precision_recall_fscore_support(y[idx], y_pred[idx],
                                                       # list the labels explicitly to always get
                                                       # the same dimension of the outputs
                                                       labels=list(assumptions.TEMPERATURE_LABEL_MAPPING.values()),
                                                       zero_division=0)
            values.append((column, lower_bound, upper_bound, idx.sum(),
                           precision_recall_fscore_support(y[idx], y_pred[idx], average="macro", zero_division=0)[2],
                           precision_recall_fscore_support(y[idx], y_pred[idx], average="weighted", zero_division=0)[2],
                           *[e for t in zip(by_label[2], by_label[3]) for e in t]  # f1 and count by label
                           ))

    table = Table("Feature", "Lower Bound", "Upper Bound", "Count", "Macro F1", "Weighted F1",
                  *[e for t in [(l, "#") for l in assumptions.TEMPERATURE_LABEL_MAPPING] for e in t],
                  title="Worst regions")
    for row in sorted(values, key=lambda x: x[4])[:top_k]:
        table.add_row(*(f"{e:.3g}" if not isinstance(e, str) else e for e in row))

    console = Console()
    console.print(table)


def find_problematic_entries(clf: CalibratedClassifierCV, x: pd.DataFrame, y: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """Find points where the predicted probabilities are most wrong

    Returns dataframe which contains the data from y and y_pred as well as the predicted probability
    of the true class

    Args:
        x (pd.DataFrame): the features
        y (pd.Series): the ground truth labels
        y_pred (np.ndarray): the predicted labels
    """
    probabilities = clf.predict_proba(x)
    ret = pd.concat((x, y), axis=1)
    ret = pd.concat((ret, pd.Series(y_pred, y.index, name="comfort_int_pred")), axis=1)
    ret["probabilities"] = probabilities.tolist()
    # maps the comfort label to position in probabilities
    comfort_int_mapping = {key: i for i, key in enumerate(assumptions.TEMPERATURE_LABEL_MAPPING.values())}
    col_idx = ret["comfort_int"].map(comfort_int_mapping).to_numpy()
    ret["probability_true_label"] = probabilities[np.arange(len(ret)), col_idx]

    worst_entries = ret.loc[ret["probability_true_label"] < 0.1].sort_index()
    for id in sorted({e for e in worst_entries.index if e not in assumptions.IDS_CHECKED}):
        webbrowser.open_new_tab(f"https://runalyze.com/activity/{id}/edit")
        print(worst_entries.loc[id])
        pyperclip.copy(id)
        input("Press enter to continue")

    return ret

def weight_dates(x: pd.Series, minimum_weight=0.0, maximum_weight=1) -> pd.Series:
    """Assigns a weight to each date, scaled linearly between minimum_weight and maximum_weight.

    Args:
        x (pd.Series): the dates
        minimum_weight (float, optional): the minimum weight, i.e. the weight of the first date. Defaults to 0.5.
        maximum_weight (float, optional): the maximum weight, i.e. the weight of the last date. Defaults to 1.

    Returns:
        np.ndarray: the weights
    """
    dates = pd.to_datetime(x, utc=True)
    minimum_date = x.min()
    total_span = x.max() - minimum_date
    return minimum_weight + (maximum_weight - minimum_weight) * (dates - minimum_date) / total_span
