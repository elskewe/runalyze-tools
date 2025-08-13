"""This module contains the assumptions used for the data."""
from datetime import datetime, timezone
from functools import lru_cache
import json
import time
from geopy.geocoders import Nominatim

GEOPY_CACHE = "cache/geopy.json"

INVALID_CLOTHING = [
    "Kompressionsstrümpfe",  # only for injury reasons
    "Trainingshose",  # only for warm up and not representative
    "Trainingsjacke",  # only for warm up and not representative
    # too little data for the following items
    "Regenjacke"
]

# clothing that is only used in races
CLOTHING_ONLY_FOR_RACES = [
    "Armwärmer",
    "Unterziehshirt mit kurzen Ärmeln"
]

# clothing categories sorted by warmth (from cold to warm)
SORTED_CLOTHING = {
    "lower_body": [
        "Ganz kurze Hose",
        "Kurze Hose",
        "Kurze Tights",
        "Lange Tights"
    ],
    "upper_body": [
        "Oberkörperfrei",
        "Singlet",
        "T-Shirt",
        "Langarmshirt",
        "Gefüttertes Langarmshirt",
        "Laufjacke"
    ],
    "hands": [
        "Handschuhe",
        "Skihandschuhe"
    ],
    "neck": [
        "Halstuch",
        "Schal",
        "Sturmhaube"
    ],
    "head": [
        "Stirnband",
        "Mütze",
    ]
}

ALL_CLOTHING_ITEMS = [e for v in SORTED_CLOTHING.values() for e in v]


# the key is replaced by the value
CLOTHING_REPLACEMENTS = {
    "2. Langarmshirt": "Langarmshirt"  # only because Runalyze can't handle multiple entries of the same item
}


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


def get_location(date: datetime, recurring_route: dict) -> tuple[str, float, float]:
    """Tries to guess the location.

    If there is a recurring route, this is used, otherwise the location is guessed based on the date.

    Args:
        date (datetime): The date of the activity.
        recurring_route (dict): The recurring route dict.

    Returns:
        tuple[str, float, float]: The name, latitude and longitude.
    """
    if recurring_route and recurring_route.get("name"):
        return (recurring_route["name"],) + lookup_location(recurring_route["name"])
    else:
        return "", 0, 0


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
    if location:
        locations[name] = {"address": location.address, "latitude": location.latitude, "longitude": location.longitude}
    else:
        locations[name] = {"address": None, "latitude": None, "longitude": None}
    with open(GEOPY_CACHE, "w", encoding="utf-8") as f:
        json.dump(locations, f, indent=4)
    time.sleep(1)  # to satisfy the rate limit in ToS
    return locations[name]["latitude"], locations[name]["longitude"]  # type: ignore


def convert_equipment(equipment: list[dict[str, str | int]]) -> list[str]:
    """Converts the equipment to a list of strings.

    Also filters the data by removing everything that is not a clothing item and applying the
    specified conversions
    """
    # apply replacements and remove everything that is not a clothing item
    return [CLOTHING_REPLACEMENTS.get(name, name) for d in equipment
            if (name := d.get("name")) in ALL_CLOTHING_ITEMS or name in INVALID_CLOTHING]


def split_equipment(equipment: list[str]) -> dict[str, list[str]]:
    """Splits the equipment into categories.

    Args:
        equipment (list[str]): The equipment as list of strings.

    Returns:
        dict[str, list[str]]: The equipment split into categories. The keys are the categories and
                              the values are the items in the category.
    """
    return {k: [e for e in equipment if e in v] for k, v in SORTED_CLOTHING.items()}
