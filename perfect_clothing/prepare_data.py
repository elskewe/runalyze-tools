"""Prepares the data starting from the json data to a pandas dataframe with several fixes applied.

Furthermore, the clothing is encoded for use with ML.
"""

import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from perfect_clothing import assumptions, weather
from runalyze import api


def prepare_data(data: list[api.ActivityType]) -> tuple[pd.DataFrame, list[str]]:
    """Prepares the data by converting the json data to a pandas dataframe.

    Also adds encoding for clothing layers etc. and returns the names of these new columns.
    """
    data_df = convert_to_df(data)
    data_df = remove_invalid_data(data_df)
    data_df = remove_unused_columns(data_df)
    data_df["pace"] = data_df["distance"]/(data_df["duration"]/3600)
    data_df["variability_index_pace"] = data_df["x_pace"] / data_df["pace"]
    data_df["x_pace_squared"] = data_df["x_pace"] ** 2
    data_df["stopped_time"] = data_df["elapsed_time"] - data_df["duration"]
    # recorded cloud cover is not reliable
    data_df["cloud_cover"] = data_df["weather_condition"].apply(assumptions.get_cloud_cover)
    data_df["wind_chill"] = data_df.apply(lambda row: weather.wind_chill(row["temperature"], row["wind_speed"]), axis=1)
    data_df[["ghi_start", "ghi_middle", "ghi_end"]] = data_df.groupby(
        # grouping by timezone is necessary to construct a pd.DatetimeIndex object
        ["latitude", "longitude", "timezone_offset"],
        sort=False, group_keys=False).apply(get_radiation_data, include_groups=False)
    data_df = data_df.dropna(subset=assumptions.INPUT_COLUMNS.keys()) # type: ignore
    data_df, encoded_clothing_columns = encode_clothing_layers(data_df)
    data_df, encoded_clothing_columns = clean_data(data_df, encoded_clothing_columns)
    data_df["comfort_int"] = data_df["comfort"].map(assumptions.TEMPERATURE_LABEL_MAPPING)
    data_df["weather_condition_int"] = data_df["weather_condition"].map(assumptions.WEATHER_CONDITION_MAPPING.index)
    data_df["is_augmented"] = False
    data_df["note_sentiment"] = data_df.apply(assumptions.get_note_sentiment, axis=1)
    print("Created pandas df")
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

        tags = tuple(tag["tag"] for tag in e.get("tags", [])) if isinstance(e.get("tags"), list) else ()  # type: ignore
        e["tags"] = tags # type: ignore

        # Extract comfort label from tags
        comfort_tags = [tag for tag in tags if assumptions.TEMPERATURE_LABEL_MAPPING.get(tag) is not None]
        if len(comfort_tags) > 1:
            print(f"Activity {e['id']} has multiple comfort tags: {comfort_tags}.")
            # make this activity invalid as it is not clear which tag is correct
            e["equipment"] = None  # type: ignore
        e["comfort"] = comfort_tags[0] if comfort_tags else assumptions.OK_TEMPERATURE_LABEL

        if "laufbahn" in tags:
            e["is_track"] = True

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


def remove_unused_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Drops columns that are not required for the clothing prediction workflow."""
    return data.drop(columns=assumptions.UNUSED_COLUMNS, errors="ignore")


def clean_data(data: pd.DataFrame, encoded_clothing_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Cleans the data

    - fixes the case that "zuHeiss" is set despite being able to shed another layer
    - removes outfits that only occur infrequently (except those used in races)
        - then removes clothing encoding columns which are now superfluous
    - removes unused columns
    """
    # When "zuHeiss" is set despite being able to shed another layer change the label to
    # "zuWarmAngezogen"
    data.loc[(~data.apply(lambda r: assumptions.can_outfit_be_toHeiss(r[encoded_clothing_columns], r["is_race"]), axis=1))
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
    return data, encoded_clothing_columns


def get_radiation_data(data: pd.DataFrame) -> pd.DataFrame:
    """Returns the radiation data for a given data frame.

    The data frame must be grouped by the latitude and longitude (in this order!) as these values
    are read from `data.name`."""
    with warnings.catch_warnings():
        # not worth preventing them as pandas datetime only works in UTC and thus would have to be
        # converted back to tz-aware immediately afterwards
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

        latitude = data.name[0]
        longitude = data.name[1]

        # contains all `ghi_start`, then all `ghi_middle` and finally all `ghi_end`
        radiation_data = weather.get_radiation(
            latitude, longitude,
            pd.DatetimeIndex(pd.concat((data["date_time"],
                                        data["date_time"] + pd.to_timedelta(data["elapsed_time"]/2, unit="s"),
                                        data["date_time"] + pd.to_timedelta(data["elapsed_time"],   unit="s")))),
            np.tile(data["cloud_cover"].to_numpy(), 3)
        )

        ghi_start, ghi_middle, ghi_end = np.split(radiation_data, 3)

        return pd.DataFrame({"ghi_start": ghi_start, "ghi_middle": ghi_middle, "ghi_end": ghi_end}, index=data.index)


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

def save_candidate_outfits(data: pd.DataFrame, encoded_clothing_columns: list[str]) -> list[dict[str, int]]:
    """Saves the candidate outfits to a json file for later use."""
    outfits_df = data[encoded_clothing_columns].value_counts().reset_index(name="n_worn")
    # I don't like that these conditions are very similar to those in `clean_data`, but they are not
    # quite the same s.t. moving them to a function is hard and the available data is just not the
    # same here
    outfits_df["race_only"] = ((outfits_df["n_worn"] < assumptions.OUTFIT_FREQUENCY_THRESHOLD)
                               & ~(outfits_df.apply(assumptions.outfit_to_string, axis=1)
                                   .str.contains("|".join(assumptions.OUTFIT_FREQUENCY_THRESHOLD_EXCEPTIONS))))
    outfits = outfits_df.to_dict(orient='records')

    with open(assumptions.CANDIDATE_OUTFITS_FILENAME, "w", encoding="utf-8") as f:
        json.dump(outfits, f, indent=4)

    return [{k: v for k, v in d.items() if k not in ("n_worn", "race_only")} for d in outfits]  # type: ignore
