"""Contains functions related to weather (e.g. sun radiation)."""

from datetime import datetime
from typing import cast, TypeVar
import pvlib
import pandas as pd
import numpy as np

NumericType = TypeVar("NumericType", float, np.ndarray)


def get_radiance(latitude: float, longitude: float, dates: pd.DatetimeIndex | datetime, 
                 cloud_cover: np.ndarray) -> np.ndarray:
    """Returns the sun radiation for a given location and dates.

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

    location = pvlib.location.Location(latitude, longitude)
    # cast is valid assuming a scalar input (instead of a pandas series)
    clearsky = cast(pd.DataFrame, location.get_clearsky(dates))
    ghi = cloud_cover_to_ghi_linear(cloud_cover, clearsky["ghi"].to_numpy())
    return ghi


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
    ghi = (offset + (1 - offset) * (1 - cloud_cover)) * ghi_clear
    return ghi
