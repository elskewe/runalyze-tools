from typing import cast
import pandas as pd
from perfect_clothing import load_data, weather, assumptions
from runalyze import api
from datetime import datetime


def train():
    # basic steps:
    # 3. add radiance data (including cloud cover)
    #    3. adjust time to reflect activity duration
    # 4. compute apparent temperature? Ask ChatGPT
    # 5. train (steps 3)-6) in https://chatgpt.com/c/689769a2-3eb8-832b-acdc-4423c037fa03, needs more clarification. Maybe ask non-reasoning model a similar prompt?)

    data = load_data.get_data()
    data_df = convert_to_df(data)
    data_df = clean_data(data_df)
    # recorded cloud cover is not reliable
    data_df["cloud_cover"] = data_df["weather_condition"].apply(assumptions.get_cloud_cover)
    data_df["ghi"] = data_df.groupby(
        # grouping by timezone is necessary to construct a pd.DatetimeIndex object
        ["latitude", "longitude", "timezone_offset"],
        sort=False, group_keys=False).apply(get_radiation_data)
    data_df


def convert_to_df(data: list[api.ActivityType]) -> pd.DataFrame:
    """Converts the json data to a pandas dataframe."""
    for e in data:
        e["sport"] = e["sport"]["name"]  # type: ignore
        if isinstance(t := e.get("type"), dict):
            e["type"] = t["name"]
        e["date_time"] = datetime.fromisoformat(e["date_time"])  # type: ignore
        e["location"], e["latitude"], e["longitude"] = (
            assumptions.get_location(e["date_time"], e.get("recurring_route")))  # type: ignore

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
    return data


def get_radiation_data(data: pd.DataFrame) -> pd.Series:
    """Returns the radiation data for a given data frame. Assumes that the location is identical for each row."""
    return pd.Series(weather.get_radiation(
        data["latitude"].iloc[0], data["longitude"].iloc[0],
        pd.to_datetime(data["date_time"].to_list()), data["cloud_cover"].to_numpy()),
        index=data.index)
