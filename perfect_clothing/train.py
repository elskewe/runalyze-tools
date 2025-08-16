from datetime import datetime, timedelta
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import lightgbm as lgb
import matplotlib.pyplot as plt

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

    base = lgb.LGBMClassifier(n_estimators=n_estimators, num_leaves=195, verbose=-1)
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

    # save model
    with open(assumptions.MODEL_FILENAME, "wb") as f:
        pickle.dump(clf, f)
    return None


def prepare_data(data: list[api.ActivityType]) -> tuple[pd.DataFrame, list[str]]:
    """Prepares the data by converting the json data to a pandas dataframe.

    Also adds encoding for clothing layers etc. and returns the names of these new columns.
    """
    data_df = convert_to_df(data)
    data_df = clean_data(data_df)
    # recorded cloud cover is not reliable
    data_df["cloud_cover"] = data_df["weather_condition"].apply(assumptions.get_cloud_cover)
    data_df["wind_chill"] = data_df.apply(lambda row: weather.wind_chill(row["temperature"], row["wind_speed"]), axis=1)
    data_df["ghi"] = data_df.groupby(
        # grouping by timezone is necessary to construct a pd.DatetimeIndex object
        ["latitude", "longitude", "timezone_offset"],
        sort=False, group_keys=False).apply(get_radiation_data)
    data_df, encoded_clothing_columns = encode_clothing_layers(data_df)
    data_df["comfort_int"] = data_df["comfort"].map(assumptions.TEMPERATURE_LABEL_MAPPING)
    data_df["weather_condition_int"] = data_df["weather_condition"].map(assumptions.WEATHER_CONDITION_MAPPING)
    return data_df, encoded_clothing_columns


def convert_to_df(data: list[api.ActivityType]) -> pd.DataFrame:
    """Converts the json data to a pandas dataframe."""
    for e in data:
        e["sport"] = e["sport"]["name"]  # type: ignore
        if isinstance(t := e.get("type"), dict):
            e["type"] = t["name"]
        e["date_time"] = (datetime.fromisoformat(e["date_time"])  # type: ignore
                          # move time to approximate middle of activity
                          + timedelta(seconds=e.get("elapsed_time", e.get("duration"))/2))  # type: ignore
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

    return pd.DataFrame(data)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
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
    # remove data with outfits that only occur infrequently
    s = data['equipment'].apply(tuple)  # convert list to tuple
    data = data[s.map(s.value_counts()) >= assumptions.OUTFIT_FREQUENCY_THRESHOLD]
    return data


def get_radiation_data(data: pd.DataFrame) -> pd.Series:
    """Returns the radiation data for a given data frame. Assumes that the location is identical for each row."""
    return pd.Series(weather.get_radiation(
        data["latitude"].iloc[0], data["longitude"].iloc[0],
        pd.to_datetime(data["date_time"].to_list()), data["cloud_cover"].to_numpy()),
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

    for _, current_row in data.iterrows():
        new_rows.extend(generate_new_outfit(current_row, encoded_clothing_columns, candidate_outfits))

    # add to base dataframe
    augmented = pd.concat([data, pd.DataFrame(new_rows)])

    return augmented


def generate_new_outfit(base_row: pd.Series, encoded_clothing_columns: list[str],
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
            | {"comfort": new_label, "comfort_int": assumptions.TEMPERATURE_LABEL_MAPPING[new_label]}
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
