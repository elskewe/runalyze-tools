import json
from operator import itemgetter
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any
import webbrowser

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
import pyperclip
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from perfect_clothing import assumptions, load_data, weather
from runalyze import api


def train():
    try:
        # delete the cache file if it is older than the raw data (jumping to the catch if the file
        # is not found is find as it won't be possible to load it either)
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

    train_core(data_df, encoded_clothing_columns, n_estimators=31)


def train_core(data: pd.DataFrame, encoded_clothing_columns: list[str], n_estimators=-1):
    """Actually train the model.

    n_estimator is passed to LGBMClassifier. If it is -1, it will be optimized beforehand by a
    hyperparameter search.
    """
    x = data[[*assumptions.INPUT_COLUMNS, *encoded_clothing_columns]]
    y = data["comfort_int"]
    train_idx, test_idx = next(
        StratifiedGroupKFold(shuffle=True).split(x, y, data.index))
    x_train, x_test, y_train, y_test = x.iloc[train_idx], x.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    x_weights = weight_dates(data["date_time"]).iloc[train_idx]

    # this is a list instead of a generator to allow pickle and use in both the hyperparameter
    # search and actual training
    cv = list(StratifiedGroupKFold(shuffle=True).split(x_train, y_train, groups=data.index[train_idx]))

    base = lgb.LGBMClassifier(n_estimators=n_estimators, num_leaves=10,
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
        search.fit(x_train, y_train, sample_weight=x_weights)
        base = search.best_estimator_
        print(search.best_params_)

    # Calibrate for better probabilities (sigmoid works reliably with moderate data)
    clf = CalibratedClassifierCV(base, cv=cv, method='sigmoid')

    clf.fit(x_train, y_train, sample_weight=x_weights)

    y_pred = clf.predict(x_test)
    labels = list(assumptions.TEMPERATURE_LABEL_MAPPING.keys())
    print(classification_report(y_test, y_pred, target_names=labels))
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()

    plot_importances(clf, x.columns.to_list())
    find_problematic_regions(x_test, y_test, y_pred)
    # just calling this to set a debugger breakpoint is not nice, but I don't want to think of a
    # good output right now
    problematic_entries = find_problematic_entries(clf, x_test, y_test, y_pred)

    # save model
    with open(assumptions.MODEL_FILENAME, "wb") as f:
        pickle.dump(clf, f)
    return None


def prepare_data(data: list[api.ActivityType]) -> tuple[pd.DataFrame, list[str]]:
    """Prepares the data by converting the json data to a pandas dataframe.

    Also adds encoding for clothing layers etc. and returns the names of these new columns.
    """
    data_df = convert_to_df(data)
    data_df = remove_invalid_data(data_df)
    data_df["x_pace_squared"] = data_df["x_pace"] ** 2
    # recorded cloud cover is not reliable
    data_df["cloud_cover"] = data_df["weather_condition"].apply(assumptions.get_cloud_cover)
    data_df["wind_chill"] = data_df.apply(lambda row: weather.wind_chill(row["temperature"], row["wind_speed"]), axis=1)
    data_df[["ghi_start", "ghi_middle", "ghi_end"]] = data_df.groupby(
        # grouping by timezone is necessary to construct a pd.DatetimeIndex object
        ["latitude", "longitude", "timezone_offset"],
        sort=False, group_keys=False).apply(get_radiation_data)
    data_df, encoded_clothing_columns = encode_clothing_layers(data_df)
    data_df, encoded_clothing_columns = clean_data(data_df, encoded_clothing_columns)
    data_df["comfort_int"] = data_df["comfort"].map(assumptions.TEMPERATURE_LABEL_MAPPING)
    data_df["weather_condition_int"] = data_df["weather_condition"].map(assumptions.WEATHER_CONDITION_MAPPING.index)
    return data_df, encoded_clothing_columns


def convert_to_df(data: list[api.ActivityType]) -> pd.DataFrame:
    """Converts the json data to a pandas dataframe."""
    for e in data:
        e["sport"] = e["sport"]["name"]  # type: ignore
        if isinstance(t := e.get("type"), dict):
            e["type"] = t["name"]
        e["date_time"] = datetime.fromisoformat(e["date_time"])  # type: ignore
        e["location"], e["latitude"], e["longitude"] = (
            assumptions.get_location(e["date_time"], e.get("recurring_route")))  # type: ignore
        if isinstance(equipment := e.get("equipment"), list):
            e["equipment"] = assumptions.convert_equipment(equipment)  # type: ignore
            e |= assumptions.split_equipment(e["equipment"])
        e["is_race"] = bool(e.get("race_result"))
        # if there is a comfort label use it, otherwise use ok
        e["comfort"] = next(
            (tag["tag"]
             for tag in e.get("tags", [])  # type: ignore
             if assumptions.TEMPERATURE_LABEL_MAPPING.get(tag["tag"]) is not None),
            assumptions.OK_TEMPERATURE_LABEL
        )
        if isinstance(tags := e.get("tags"), list):
            e["tags"] = tuple(tag["tag"] for tag in tags)  # type: ignore

    # try filling missing location data
    for i, e in enumerate(data):
        if e["latitude"] == 0 and e["longitude"] == 0:
            # use older data point (list starts with newest activities)
            e["latitude"] = data[i + 1]["latitude"]
            e["longitude"] = data[i + 1]["longitude"]
            if e["latitude"] == 0 and e["longitude"] == 0:
                # use newer data point as alternative
                e["latitude"] = data[i - 1]["latitude"]
                e["longitude"] = data[i - 1]["longitude"]

    return pd.DataFrame.from_records(data, index="id")


def remove_invalid_data(data: pd.DataFrame) -> pd.DataFrame:
    """Removes activities with missing data"""
    data = data[~data["weather_condition"].str.contains("unknown")]  # missing weather
    data = data.dropna(subset=["equipment"])  # without equipment data the activity is not usable
    # remove activities with invalid clothing items
    data = data[~data["equipment"].apply(lambda lst: any(e in assumptions.INVALID_CLOTHING for e in lst))]
    # remove activities without a value set for either lower body or upper body clothing
    data = data[(data["lower_body"].str.len() > 0) & (data["upper_body"].str.len() > 0)]
    # remove activities with race only clothing which are not races
    data = data[data["is_race"]
                | ~data["equipment"].apply(lambda lst: any(e in assumptions.CLOTHING_ONLY_FOR_RACES for e in lst))]
    return data  # noqa: RET504


def clean_data(data: pd.DataFrame, encoded_clothing_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Cleans the data

    - fixes the case that "zuHeiss" is set despite being able to shed another layer
    - removes outfits that only occur infrequently (except those used in races)
        - then removes clothing encoding columns which are now superfluous
    - removes unused columns
    """
    # When "zuHeiss" is set despite being able to shed another layer change the label to
    # "zuWarmAngezogen"
    # 1. for non-races
    data.loc[(data["upper_body_layer1"] > assumptions.SORTED_CLOTHING["upper_body"].index("Oberkörperfrei")+1)
             & (data["comfort"] == "zuHeiss") & ~data["is_race"], "comfort"] = "zuWarmAngezogen"
    # 2. for races
    data.loc[(data["upper_body_layer1"] > assumptions.SORTED_CLOTHING["upper_body"].index("Singlet")+1)
             & (data["comfort"] == "zuHeiss") & data["is_race"], "comfort"] = "zuWarmAngezogen"
    # 3. for lower body
    data.loc[(data["lower_body_layer1"] > assumptions.SORTED_CLOTHING["lower_body"].index("Ganz kurze Hose")+1)
             & (data["comfort"] == "zuHeiss"), "comfort"] = "zuWarmAngezogen"

    # Remove data with outfits that only occur infrequently. An exception is made for combinations
    # which are valid for races (i.e. the have the necessary equipment from
    # `assumptions.NECESSARY_EQUIPMENT_FOR_RACES`) because otherwise there is too little data for
    # these outfits as well as items specified in
    # `assumptions.OUTFIT_FREQUENCY_THRESHOLD_EXCEPTIONS`.
    e = data["equipment"]
    data = data[(e.map(e.value_counts()) >= assumptions.OUTFIT_FREQUENCY_THRESHOLD)
                | (data[encoded_clothing_columns].apply(
                    lambda r: assumptions.valid_outfits([r.to_dict()], True), axis=1)
                    # The return is either the (valid) outfit or an empty list. This truthy and
                    # can thus be converted to bool directly
                    .apply(bool))
                | e.apply(lambda x: any(set(x).intersection(assumptions.OUTFIT_FREQUENCY_THRESHOLD_EXCEPTIONS)))]
    # remove clothing encoding columns which are now superfluous due to not having any outfits with
    # that many layers
    columns_to_remove = [c for c in encoded_clothing_columns if (data[c] == 0).all()]
    data = data.drop(columns=columns_to_remove)
    encoded_clothing_columns = [c for c in encoded_clothing_columns if c not in columns_to_remove]
    # remove unused columns to reduce pickle size
    data = data.drop(columns=["recurring_route", "downhill_efficiency", "uphill_efficiency", "device_id",
                              "required_critical_power", "required_critical_pace", "fit_sweat_loss", "wheel_size",
                              "required_critical_pace_vo2max", "fit_hrv_analysis", "jumps", "total_strokes", "swolf",
                              "ozone"])
    return data, encoded_clothing_columns


def get_radiation_data(data: pd.DataFrame) -> pd.DataFrame:
    """Returns the radiation data for a given data frame. Assumes that the location is identical for each row."""
    with warnings.catch_warnings():
        # not worth preventing them as pandas datetime only works in UTC and thus would have to be
        # converted back to tz-aware immediately afterwards
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

        def get_radiation(time: pd.Series) -> np.ndarray:
            return weather.get_radiation(data["latitude"].iloc[0], data["longitude"].iloc[0],
                                         # pylance somehow thinks this is a `Index[int]`
                                         pd.DatetimeIndex(time),  # type: ignore
                                         data["cloud_cover"].to_numpy())

        return pd.DataFrame({"ghi_start": get_radiation(data["date_time"]),
                             "ghi_middle": get_radiation(data["date_time"]
                                                         + pd.to_timedelta(data.get("elapsed_time",  # type: ignore
                                                                                    data.get("duration"))/2, unit="s")),
                             "ghi_end": get_radiation(data["date_time"]
                                                      + pd.to_timedelta(data.get("elapsed_time",  # type: ignore
                                                                                 data.get("duration")), unit="s"))},
                            index=data.index)


def encode_clothing_layers(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Encodes the clothing layers into ints for each category.

    Also returns the added column names."""
    new_columns = []
    for category, items in assumptions.SORTED_CLOTHING.items():
        n_layers = int(data[category].str.len().max())  # number of layers needed
        column_names = [f"{category}_layer{i}" for i in range(1, n_layers + 1)]
        new_columns.extend(column_names)
        # clothing sorted descending from warmer to less warm
        encoded_clothing = data[category].apply(lambda e: sorted(e, key=items.index, reverse=True)).apply(
            # encode clothing as layers, with 0 meaning nothing in this layer and a number mapping
            # to the 1-based index in `items`
            lambda clothes: [items.index(clothes[i])+1 if i < len(clothes) else 0 for i in range(n_layers)])
        data[column_names] = pd.DataFrame(encoded_clothing.to_list(), index=data.index)

    return data, new_columns


def augment_data(data: pd.DataFrame, encoded_clothing_columns: list[str],
                 candidate_outfits: list[dict[str, int]]) -> pd.DataFrame:
    """Adds more examples where the label is not ok.

    Basic idea: if an outfit is too warm, every outfit that only contains the same or warmer items
    will also be too warm. The same principle holds for the other labels.

    Furthermore, to balance the categories afterwards, the other features might be modified as well
    to get more data for less frequent categories.
    """
    new_rows = []

    for id_, current_row in data.iterrows():
        new_rows.extend(generate_new_outfit(id_, current_row, encoded_clothing_columns, candidate_outfits))

    for factor in range(1, assumptions.MAX_AUGMENTATION_FACTOR+1):
        # get comfort labels so far
        comfort_int = pd.concat([data["comfort_int"], pd.Series([e["comfort_int"] for e in new_rows])])
        # the return of `value_counts` is sorted by frequency, thus the first value of the index is
        # always the most frequent label
        most_frequent_comfort_label = int(comfort_int.value_counts().index[0])  # type: ignore
        least_frequent_comfort_label = comfort_int.value_counts().index[-1]
        if least_frequent_comfort_label == assumptions.TEMPERATURE_LABEL_MAPPING[assumptions.OK_TEMPERATURE_LABEL]:
            break  # continue this loop until `ok` is the least frequent label (it can't be augmented)
        for id_, current_row in data.iterrows():
            new_rows.extend(generate_new_features(id_, current_row, most_frequent_comfort_label, factor))

    for id_, current_row in data.iterrows():
        new_rows.extend(generate_new_features_ok(id_, current_row, assumptions.MAX_AUGMENTATION_FACTOR))

    # add to base dataframe
    return pd.concat([data, pd.DataFrame.from_records(new_rows, index="id")])


def generate_new_outfit(id_: int, base_row: pd.Series, encoded_clothing_columns: list[str],
                        candidate_outfits: list[dict[str, int]]) -> list[dict]:
    """Generates new outfits based on the given label (`base_row["comfort_int"]`) and note.

    The new outfits are warmer/colder (depending on the label) in at least one clothing category and
    at least the same in the rest, which means the direction of the comfort (too cold/too warm) is
    retained. The new outfit (with the other data from the base_row) is than added to the return
    value. Besides the actual label, an attempt is made to determine the sentiment of the note. This
    sentiment is used like the actual label if they do not contradict each other. The idea here is
    the same, if an outfit is almost too warm, then adding items will finally push that outfit into
    being too warm (the same principle applies for too cold).

    Args:
        base_row (pd.Series): the row from which the new outfits are generated
        encoded_clothing_columns (list[str]): the names of the clothing columns candidate_outfits
        (list[dict[str, int]]): the possible outfits

    Returns:
        list[dict]: the additional rows with the altered outfits
    """
    outfit = base_row[encoded_clothing_columns].to_dict()
    # The sentiment of the note is "zuWarmAngezogen" (`1`) if it contains any word from
    # `ALMOST_TOO_WARM_WORDS` and "zuKaltAngezogen" (`-1`) if it contains any word from
    # `ALMOST_TOO_COLD_WORDS`. If words from both list or none are present, the note is not used as
    # it is ambiguous.
    note_sentiment = (any(w in base_row["note"] for w in assumptions.ALMOST_TOO_WARM_WORDS)
        - any(w in base_row["note"] for w in assumptions.ALMOST_TOO_COLD_WORDS)) \
            if isinstance(base_row["note"], str) else 0
    label = base_row["comfort_int"]
    # If the inferred label from the note and the actual label contradict each other disregard the
    # note.
    if label * note_sentiment < 0:
        note_sentiment = 0

    if label > 0 or note_sentiment > 0:  # zuWarmAngezogen
        # Generate outfits warmer compared to this one
        new_label = "zuWarmAngezogen"
        new_rows = [candidate_outfit for candidate_outfit in candidate_outfits
                    if all(v >= outfit[k] for k, v in candidate_outfit.items()) and outfit != candidate_outfit
                        and sum(candidate_outfit.values()) - sum(outfit.values()) in assumptions.MAX_CLOTHING_DISTANCE]
    elif label < 0 or note_sentiment < 0:  # zuKaltAngezogen
        # Generate outfits cooler compared to this one
        new_label = "zuKaltAngezogen"
        new_rows = [candidate_outfit for candidate_outfit in candidate_outfits
                    if all(v <= outfit[k] for k, v in candidate_outfit.items()) and outfit != candidate_outfit
                    and sum(candidate_outfit.values()) - sum(outfit.values()) in assumptions.MAX_CLOTHING_DISTANCE]
    else:  # label == 0
        return []  # no new rows are added

    return [base_row.to_dict() | new_row
            | {"comfort": new_label, "comfort_int": assumptions.TEMPERATURE_LABEL_MAPPING[new_label],
               "id": id_}
            for new_row in new_rows]


def generate_new_features(id_: int, base_row: pd.Series, most_frequent_comfort_label: int,
                          augmentation_factor: int) -> list[dict]:
    """Moves each of the features (by `augmentation_factor`) to get additional rows with the same label.

    To combat label imbalance, this is only done for labels which are not the most frequent one."""
    # Return if the label is `ok` as it's not possible to know which features to move to still
    # retain the same label. Also don't make the imbalance worse by adding new rows of the already
    # most frequent label.
    if base_row["comfort_int"] == assumptions.TEMPERATURE_LABEL_MAPPING[assumptions.OK_TEMPERATURE_LABEL] \
            or base_row["comfort_int"] == most_frequent_comfort_label:
        return []

    new_rows = []
    # amplitude and direction in which the features are moved
    factor = augmentation_factor * np.sign(base_row["comfort_int"])
    for feature, direction in assumptions.INPUT_COLUMNS.items():
        if direction == 0:
            continue
        match feature:
            case "wind_speed" | "temperature":  # also need to adapt windchill
                if feature == "wind_speed":
                    new_wind_speed = base_row["wind_speed"] + direction * factor
                    new_temperature = base_row["temperature"]
                else:  # feature == "temperature":
                    new_wind_speed = base_row["wind_speed"]
                    new_temperature = base_row["temperature"] + direction * factor
                if new_wind_speed < 0:
                    continue  # invalid wind speed
                new_rows.append({"wind_speed": new_wind_speed,
                                 "temperature": new_temperature,
                                 "wind_chill": weather.wind_chill(new_temperature, new_wind_speed)})
            case "ghi_start":
                # I don't like the hardcoded value here, but it's the easiest way right now
                new_cloud_cover_perc = base_row["cloud_cover"] - 50 * factor
                if new_cloud_cover_perc < 0 or new_cloud_cover_perc > 100:
                    continue  # cloud cover is already at minimum or maximum
                tmp_df = pd.DataFrame([base_row])
                tmp_df["cloud_cover"] = new_cloud_cover_perc
                # the `group` is only to get the correct type
                tmp_df[["ghi_start", "ghi_middle", "ghi_end"]] = \
                    tmp_df.groupby(["latitude"], group_keys=False).apply(get_radiation_data)
                new_rows.append({"cloud_cover": new_cloud_cover_perc,
                                 "ghi_start": tmp_df["ghi_start"].iloc[0],
                                 "ghi_middle": tmp_df["ghi_middle"].iloc[0],
                                 "ghi_end": tmp_df["ghi_end"].iloc[0]})
            case "ghi_middle" | "ghi_end":
                continue  # already handled by `ghi_start`
            case "x_pace":
                new_x_pace = base_row["x_pace"] + direction * factor
                if new_x_pace < 0:
                    continue  # invalid x_pace
                new_rows.append({feature: new_x_pace})
            case _:
                new_rows.append({feature: base_row[feature] + direction * factor})

    return [base_row.to_dict() | new_row | {"id": id_} for new_row in new_rows]

def generate_new_features_ok(id_: int, base_row: pd.Series, augmentation_factor: int) -> list[dict]:
    """Moves the temperature (by `augmentation_factor`) to get additional rows with the label changed correspondingly.

    Only looks at rows where the temperature label is ok. To get reasonable results, the augmentation factor should be
    high enough that the new features definitely change the label."""

    if base_row["comfort_int"] != assumptions.TEMPERATURE_LABEL_MAPPING[assumptions.OK_TEMPERATURE_LABEL]:
        return []

    new_rows = []
    # amplitude and direction in which the features are moved
    for new_comfort_int in itemgetter("zuKaltAngezogen", "zuWarmAngezogen")(assumptions.TEMPERATURE_LABEL_MAPPING):
        factor = augmentation_factor * new_comfort_int
        for feature, direction in assumptions.INPUT_COLUMNS.items():
            if direction == 0:
                continue
            match feature:
                case "temperature":  # also need to adapt windchill
                    new_temperature = base_row["temperature"] + direction * factor
                    new_rows.append({"temperature": new_temperature,
                                    "wind_chill": weather.wind_chill(new_temperature, base_row["wind_speed"]),
                                    "comfort_int": new_comfort_int})
                case _:  # it's not certain enough that the new features change the label
                    continue

    return [base_row.to_dict() | new_row | {"id": id_} for new_row in new_rows]


def save_candidate_outfits(data: pd.DataFrame, encoded_clothing_columns: list[str]) -> list[dict[str, int]]:
    """Saves the candidate outfits to a json file for later use."""
    outfits = data[encoded_clothing_columns].value_counts().reset_index(name="n_worn").to_dict(orient='records')

    with open(assumptions.CANDIDATE_OUTFITS_FILENAME, "w", encoding="utf-8") as f:
        json.dump(outfits, f, indent=4)

    return [{k: v for k, v in d.items() if k != "n_worn"} for d in outfits]  # type: ignore


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
