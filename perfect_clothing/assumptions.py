"""This module contains the assumptions used for the data."""
from datetime import datetime, timezone
from functools import lru_cache
import json
import time
from geopy.geocoders import Nominatim

GEOPY_CACHE = "cache/geopy.json"


def get_cloud_cover(condition: str) -> int:
    """Returns the cloud cover for a given weather condition.

    Args:
        condition (str): The weather condition.

    Returns:
        int: The cloud cover in percent.
    """
    match condition:
        case "sunny":
            return 0
        case "fair" | "changeable":
            return 50
        case "cloudy" | "rainy" | "heavyrain" | "snowing" | "windy" | "foggy" | "thunderstorm":
            return 100
        case _:
            raise ValueError(f"Unknown weather condition: {condition}")


def get_location(date: datetime, recurring_route: dict) -> tuple[float, float]:
    """Tries to guess the location.

    If there is a recurring route, this is used, otherwise the location is guessed based on the date.

    Args:
        date (datetime): The date of the activity.
        recurring_route (dict): The recurring route dict.

    Returns:
        tuple[float, float]: The latitude and longitude.
    """
    if date < datetime(2020, 1, 1, tzinfo=timezone.utc):
        return 48.5, 8.8
    return 47.7, 9.15  #TODO: find a better location


@lru_cache(maxsize=None)
def lookup_location(name: str) -> tuple[float, float]:
    """Tries to get the coordinates for a location name.

    Args:
        name (str): The name of the location.

    Returns:
        tuple[float, float]: The latitude and longitude.
    """
    try:
        with open(GEOPY_CACHE, "r", encoding="utf-8") as f:
            locations = json.load(f)
    except FileNotFoundError:
        locations = {}

    if name in locations:
        return locations[name]["latitude"], locations[name]["longitude"]

    geolocator = Nominatim(user_agent="elskewe-runalyze-tools")
    location = geolocator.geocode(name)
    locations[name] = {"address": location.address, "latitude": location.latitude, "longitude": location.longitude}
    with open(GEOPY_CACHE, "w", encoding="utf-8") as f:
        json.dump(locations, f, indent=4)
    time.sleep(1)  # to satisfy the rate limit in ToS
    return location.latitude, location.longitude  # type: ignore
