"""Contains functions related to weather (e.g. sun radiation)."""

from datetime import datetime
from typing import TypeVar, cast
import warnings

import numpy as np
import pandas as pd
import pvlib

from perfect_clothing.assumptions import get_shade_from_solar_elevation

NumericType = TypeVar("NumericType", float, np.ndarray)


def wind_chill(temperature: float, wind_speed: float) -> float:
    """Returns the windchill temperature.

    Args:
        temperature (float): The temperature in °C.
        wind_speed (float): The wind speed in km/h.

    Returns:
        float: The windchill temperature in °C.
    """
    if temperature > 10 or wind_speed < 4.8:
        # windchill is not defined at this conditions
        return temperature
    return 13.12 + 0.6215 * temperature - 11.37 * wind_speed ** 0.16 + 0.3965 * temperature * wind_speed ** 0.16


def ghi_mean(row: pd.Series) -> float:
    """Returns the mean GHI for a given row.

    Args:
        row (pd.Series): A row of a DataFrame containing the columns `ghi_start`, `ghi_middle` and `ghi_end`.

    Returns:
        float: The mean GHI.
    """
    return row[["ghi_start", "ghi_middle", "ghi_end"]].mean()


def ghi_raw(row: pd.Series) -> pd.Series:
    """Returns the raw GHI values for a given row.

    Args:
        row (pd.Series): A row of a DataFrame containing the columns `ghi_start`, `ghi_middle` and `ghi_end`.

    Returns:
        pd.Series: The raw GHI values.
    """
    return row[["ghi_start", "ghi_middle", "ghi_end"]]


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
        radiation_data = get_radiation(
            latitude, longitude,
            pd.DatetimeIndex(pd.concat((data["date_time"],
                                        data["date_time"] + pd.to_timedelta(data["elapsed_time"]/2, unit="s"),
                                        data["date_time"] + pd.to_timedelta(data["elapsed_time"],   unit="s")))),
            np.tile(data["cloud_cover"].to_numpy(), 3)
        )

        ghi_start, ghi_middle, ghi_end = np.split(radiation_data, 3)

        return pd.DataFrame({"ghi_start": ghi_start, "ghi_middle": ghi_middle, "ghi_end": ghi_end}, index=data.index)


def get_radiation(latitude: float, longitude: float, dates: pd.DatetimeIndex | datetime | list[datetime],
                  cloud_cover: np.ndarray) -> np.ndarray:
    """Returns the sun radiation for a given location and dates.

    Note that the time complexity is roughly O(1) for `len(dates)` less than 100 as there are high
    constant costs for initializing `clearsky`.

    Args:
        latitude (float): The latitude of the location.
        longitude (float): The longitude of the location.
        dates (pd.DatetimeIndex): The dates of the activities (or a single datetime object).
        cloud_cover (int): The cloud cover in percent.

    Returns:
        float: The sun radiation in W/m2.
    """
    # convert datetime object to pandas ´DatetimeIndex`
    if isinstance(dates, datetime):
        dates = pd.to_datetime([dates])
    elif isinstance(dates, list):
        dates = pd.to_datetime(dates)

    location = pvlib.location.Location(latitude, longitude)
    # cast is valid assuming a scalar input (instead of a pandas series)
    clearsky = cast(pd.DataFrame, location.get_clearsky(dates))
    solar_position = cast(pd.DataFrame, location.get_solarposition(dates))
    adjusted_cloud_cover = get_shade_from_solar_elevation(solar_position, cloud_cover)
    return cloud_cover_to_ghi_linear(adjusted_cloud_cover, clearsky["ghi"].to_numpy())


def cloud_cover_to_ghi_linear(cloud_cover: NumericType, ghi_clear: NumericType, offset=35) -> NumericType:
    """
    Convert cloud cover to GHI using a linear relationship. 0% cloud cover returns ghi_clear. 100%
    cloud cover returns offset*ghi_clear.

    Args:
        cloud_cover (float): Cloud cover in %.
        ghi_clear (float): GHI under clear sky conditions.
        offset (float, optional): Determines the minimum GHI. Defaults to 35.

    Returns:
        ghi (float): Cloudy sky GHI.

    References:
        Larson et. al. "Day-ahead forecasting of solar power output from photovoltaic plants in the
        American Southwest" Renewable Energy 91, 11-20 (2016).

        https://solarforecastarbiter-core.readthedocs.io/en/latest/generated/solarforecastarbiter.reference_forecasts.forecast.cloud_cover_to_ghi_linear.html
    """

    offset = offset / 100.
    cloud_cover = cloud_cover / 100.
    return (offset + (1 - offset) * (1 - cloud_cover)) * ghi_clear
