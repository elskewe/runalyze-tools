from datetime import datetime
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
import lightgbm as lgb
import matplotlib.pyplot as plt
from rich.table import Table
from rich.console import Console

from perfect_clothing import load_data, weather, assumptions
from runalyze import api


def train():
    try:
        with open(assumptions.DATA_FILENAME, "rb") as f:
            (data_df, encoded_clothing_columns) = pickle.load(f)
    except FileNotFoundError:
        data = load_data.get_data()
        data_df, encoded_clothing_columns = prepare_data(data)
        candidate_outfits = save_candidate_outfits(data_df, encoded_clothing_columns)
        data_df = augment_data(data_df, encoded_clothing_columns, candidate_outfits)

        with open(assumptions.DATA_FILENAME, "wb") as f:
            pickle.dump((data_df, encoded_clothing_columns), f)

    train_core(data_df, encoded_clothing_columns, n_estimators=500)

    return None


def train_core(data: pd.DataFrame, encoded_clothing_columns: list[str], n_estimators=-1):
    """Actually train the model.

    n_estimator is passed to LGBMClassifier. If it is -1, it will be optimized beforehand by a
    hyperparameter search.
    """
    x = data[[*assumptions.INPUT_COLUMNS, *encoded_clothing_columns]]
    y = data["comfort_int"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    base = lgb.LGBMClassifier(n_estimators=n_estimators, num_leaves=195,
                              verbose=-1, class_weight="balanced", importance_type="gain",)
    if n_estimators == -1:
        param_grid = {
            "n_estimators": np.logspace(1, 3, 10, dtype=int),
            # "num_leaves": np.logspace(2.083, 2.333, 7, dtype=int)
        }
        # search for best hyperparameters
        search = GridSearchCV(base, param_grid, scoring="f1_macro", error_score="raise")
        search.fit(x_train, y_train)
        base = search.best_estimator_

    # Calibrate for better probabilities (sigmoid works reliably with moderate data)
    clf = CalibratedClassifierCV(base, method='sigmoid')

    clf.fit(x_train, y_train)

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
        e["is_race"] = True if e.get("race_result") else False
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
    return data


def clean_data(data: pd.DataFrame, encoded_clothing_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Cleans the data

    - fixes the case that "zuHeiss" is set despite being able to shed another layer
    - removes outfits that only occur infrequently (except those used in races)
        - then removes clothing encoding columns which are now superfluous
    - removes unused columns
    """
    # When "zuHeiss" is set despite being able to shed another layer change the label to
    # "zuWarmAngezogen"
    data.loc[(data["upper_body_layer1"] > assumptions.SORTED_CLOTHING["upper_body"].index("Oberkörperfrei")+1)
             & (data["comfort"] == "zuHeiss") & ~data["is_race"], "comfort"] = "zuWarmAngezogen"
    # Remove data with outfits that only occur infrequently. An exception is made for combinations
    # which are valid for races (i.e. the have the necessary equipment from
    # `assumptions.NECESSARY_EQUIPMENT_FOR_RACES`) because otherwise there is too little data for these outfits
    e = data["equipment"]
    data = data[(e.map(e.value_counts()) >= assumptions.OUTFIT_FREQUENCY_THRESHOLD)
                | (data[encoded_clothing_columns].apply(
                    lambda r: assumptions.valid_outfits([r.to_dict()], True), axis=1)
                    # The return is either the (valid) outfit or an empty list. This truthy and
                    # can thus be converted to bool directly
                    .apply(bool))]
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
    """
    new_rows = []

    for id_, current_row in data.iterrows():
        new_rows.extend(generate_new_outfit(id_, current_row, encoded_clothing_columns, candidate_outfits))

    # add to base dataframe
    augmented = pd.concat([data, pd.DataFrame.from_records(new_rows, index="id")])

    return augmented


def generate_new_outfit(id_: int, base_row: pd.Series, encoded_clothing_columns: list[str],
                        candidate_outfits: list[dict[str, int]]) -> list[dict]:
    """Generates new outfits based on the given label (`base_row["comfort_int"]`).

    The new outfits are warmer/colder (depending on the label) in at least one clothing category and
    at least the same in the rest, which means the direction of the comfort (too cold/too warm) is
    retained. The new outfit (with the other data from the base_row) is than added to the return
    value.

    Args:
        base_row (pd.Series): the row from which the new outfits are generated
        encoded_clothing_columns (list[str]): the names of the clothing columns candidate_outfits
        (list[dict[str, int]]): the possible outfits

    Returns:
        list[dict]: the additional rows with the altered outfits
    """
    new_rows = []
    if (label := base_row["comfort_int"]) == 0:
        return new_rows  # no new rows are added

    outfit = base_row[encoded_clothing_columns].to_dict()
    if label > 0:  # zuWarmAngezogen
        # Generate outfits warmer to this one
        new_label = "zuWarmAngezogen"
        for candidate_outfit in candidate_outfits:
            if all(v >= outfit[k] for k, v in candidate_outfit.items()) and outfit != candidate_outfit:
                new_rows.append(candidate_outfit)
    elif label < 0:  # zuKaltAngezogen
        # Generate outfits cooler to this one
        new_label = "zuKaltAngezogen"
        for candidate_outfit in candidate_outfits:
            if all(v <= outfit[k] for k, v in candidate_outfit.items()) and outfit != candidate_outfit:
                new_rows.append(candidate_outfit)
    else:
        raise ValueError("Shouldn't end up here")

    return [base_row.to_dict() | new_row
            | {"comfort": new_label, "comfort_int": assumptions.TEMPERATURE_LABEL_MAPPING[new_label],
               "id": id_}
            for new_row in new_rows]


def save_candidate_outfits(data: pd.DataFrame, encoded_clothing_columns: list[str]) -> list[dict[str, int]]:
    """Saves the candidate outfits to a json file for later use."""
    d = data[encoded_clothing_columns].drop_duplicates().to_dict(orient='records')

    with open(assumptions.CANDIDATE_OUTFITS_FILENAME, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=4)

    return d  # type: ignore


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
            values.append((column, lower_bound, upper_bound, idx.sum(),
                           precision_recall_fscore_support(y[idx], y_pred[idx], average="macro", zero_division=0)[2],
                           precision_recall_fscore_support(y[idx], y_pred[idx], average="weighted", zero_division=0)[2]))

    table = Table("Feature", "Lower Bound", "Upper Bound", "Count", "Macro F1", "Weighted F1", title="Worst regions")
    for row in sorted(values, key=lambda x: x[4])[:top_k]:
        table.add_row(*(f"{e:.3g}" if not isinstance(e, str) else e for e in row))

    console = Console()
    console.print(table)
