import json
import pandas as pd
from perfect_clothing import load_data, weather, assumptions
from runalyze import api
from datetime import datetime, timedelta


def train():
    # basic steps:
    # 4. compute apparent temperature? Ask ChatGPT
    # 5. train (steps 3)-6) in https://chatgpt.com/c/689769a2-3eb8-832b-acdc-4423c037fa03, needs more clarification. Maybe ask non-reasoning model a similar prompt?)

    data = load_data.get_data()
    data_df, encoded_clothing_columns = prepare_data(data)
    save_candidate_outfits(data_df, encoded_clothing_columns)

    data_df


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


def save_candidate_outfits(data: pd.DataFrame, encoded_clothing_columns: list[str]) -> None:
    """Saves the candidate outfits to a json file for later use."""
    d = data[encoded_clothing_columns].drop_duplicates().to_dict(orient='records')

    with open(assumptions.CANDIDATE_OUTFITS_FILENAME, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=4)