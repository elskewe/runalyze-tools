"""Contains functions related to weather (e.g. sun radiation)."""

from datetime import datetime
from typing import OrderedDict, cast
import pvlib


def get_radiance(latitude: float, longitude: float, date: datetime, cloud_cover: int) -> float:
    """Returns the sun radiation for a given location and date.

    Args:
        latitude (float): The latitude of the location.
        longitude (float): The longitude of the location.
        date (datetime.datetime): The date of the activity.
        cloud_cover (int): The cloud cover in percent.

    Returns:
        float: The sun radiation in W/m2.
    """
    location = pvlib.location.Location(latitude, longitude)
    # cast is valid assuming a scalar input (instead of a pandas series)
    clearsky = cast(OrderedDict, location.get_clearsky(date))
    ghi = cloud_cover_to_ghi_linear(cloud_cover, clearsky["ghi"])
    return ghi


def cloud_cover_to_ghi_linear(cloud_cover: float, ghi_clear: float, offset=35):
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
