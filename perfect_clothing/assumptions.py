"""This module contains the assumptions used for the data."""
from datetime import datetime


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
    return 47, 9  #TODO: find a better location
