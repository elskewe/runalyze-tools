"""This module contains the assumptions used for the data."""
import json
import time
from datetime import datetime
from functools import cache

from geopy.geocoders import Nominatim

GEOPY_CACHE = "cache/geopy.json"
MODEL_FILENAME = "cache/model.pickle"
ALTERNATIVE_MODEL_FILENAME = "cache/model_small.pickle"  # alternative model
DATA_FILENAME = "cache/data.pickle"  # pandas DataFrame with the training data
CANDIDATE_OUTFITS_FILENAME = "cache/candidate_outfits.json"
# keys are the columns used as the input for the prediction, the values are the increments to move
# the outfit towards a warmer feeling (0 meaning no influence)
INPUT_COLUMNS = {
    "temperature": 2,
    "wind_chill": 0,  # is already moved by wind speed
    # "humidity", => comparatively hard to obtain and does not make that big of a difference
    "wind_speed": -5,  # => this feels like it should have an influence, but I'm not convinced it's actually the case
    # "weather_condition_int": 0,  # => makes the result slightly worse and gets little weight anyway (probably too little data)
    "duration": 0,
    ##"hr_avg",
    ##"fit_trimp",
    "x_pace": 1,
    ##"x_pace_squared": 1, # => x_pace seems to be a better predictor
    # for the ghi these are dummy values only giving the sign as these should be consistent with the time of day
    "ghi_start": 1,
    "ghi_middle": 1,
    "ghi_end": 1,
    "is_race": 0}
MAX_AUGMENTATION_FACTOR = 3  # maximum factor that is multiplied on the increments to augment the data

# different outfits are used for augmenting the data as long as they are within this distance
# (w.r.t. the sum of the int encoded clothing items) of the original outfit. The first (negative)
# value is for outfits which are too cold, the second (positive) value is for outfits which are too
# warm
MAX_CLOTHING_DISTANCE = range(-15, 5)

OUTFIT_FREQUENCY_THRESHOLD = 3  # outfits that occur less than this number of times are removed
INVALID_CLOTHING = [
    "Trainingshose",  # only for warm up and not representative
    "Trainingsjacke",  # only for warm up and not representative
    # too little data for the following items
    "Regenjacke"
]

# clothing that is only used in races
CLOTHING_ONLY_FOR_RACES = [
    "Armwärmer",
    "Singlet+Armwärmer"
]
# one of these items is required for a race
NECESSARY_RACE_CLOTHING = {"upper_body": ["Singlet", "Singlet+Armwärmer"]}


# clothing categories sorted by warmth (from cold to warm)
SORTED_CLOTHING = {
    "legs": [
        "Kompressionsstrümpfe"
    ],
    "lower_body": [
        "Ganz kurze Hose",
        "Kurze Hose",
        "Kurze Tights",
        "Lange Unterhose",
        "Lange Tights"
    ],
    "upper_body": [
        "Oberkörperfrei",
        "Singlet",
        "T-Shirt",
        "Singlet+Armwärmer",
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
    "2. Langarmshirt": "Langarmshirt",  # only because Runalyze can't handle multiple entries of the same item
    "Unterziehshirt mit kurzen Ärmeln": "T-Shirt",  # should be pretty similar warmth wise
}

# every entry in the key is removed and every entry in the value added if all entries in the key are present
CLOTHING_REPLACEMENTS_MULTI = {
    ("Singlet", "Armwärmer"): ("Singlet+Armwärmer", ),
    # probably mislabelled as this combination would be quite cold on the hands
    ("Handschuhe", "Lange Tights"): ("Skihandschuhe", "Lange Tights"),
    ("Handschuhe", "Laufjacke"): ("Skihandschuhe", "Laufjacke"),
}

TEMPERATURE_LABEL_MAPPING = {'zuKaltAngezogen': -1, 'ok': 0, 'zuWarmAngezogen': 1, 'zuHeiss': 2}
OK_TEMPERATURE_LABEL = "ok"

# The following words are matched (with `pandas.Series.str.contains`, i.e. a case-insensitive string
# which might contain regex) in note to find activities which were at the edge of being ok. If
# matches from both categories occur, they are not used as the note is ambiguous.
ALMOST_TOO_COLD_WORDS = ["kühl", "kalt"]
ALMOST_TOO_WARM_WORDS = ["warm", "heiß"]


WEATHER_CONDITION_MAPPING = [
    "snowing",
    "heavyrain",
    "thunderstorm",
    "rainy",
    "windy",
    "foggy",
    "cloudy",
    "changeable",
    "fair",
    "sunny",
]

IDS_CHECKED = { # runalyze ids which were checked for obvious errors
    1362180,
    1362221,
    1362228,
    1362232,
    1411493,
    1411501,
    1423157,
    1429178,
    1454110,
    1470366,
    1656768,
    1933669,
    2005260,
    2059576,
    2103322,
    2126089,
    2179223,
    2182074,
    2192845,
    2226570,
    2227594,
    2227598,
    2227599,
    2230239,
    2279694,
    2290729,
    2345845,
    2399761,
    2439561,
    2444574,
    2461075,
    3286690,
    3609377,
    3620645,
    3637408,
    4062034,
    4087595,
    4390318,
    4410737,
    5870365,
    5844724,
    5986658,
    8017307,
    8128342,
    8360197,
    8578520,
    8775798,
    8800593,
    9022343,
    9062243,
    9167564,
    10316469,
    10917865,
    10937816,
    11149820,
    11580303,
    11657610,
    12627128,
    13062923,
    13091386,
    13257176,
    13257180,
    13473812,
    13711578,
    13953006,
    14060210,
    14111292,
    14316276,
    14638186,
    14754972,
    14878996,
    15572916,
    15970586,
    16111014,
    16278432,
    16330730,
    16369664,
    17643514,
    18454786,
    18570974,
    21071461,
    21185162,
    21287968,
    21476974,
    21558939,
    21796471,
    22007846,
    23793350,
    23829405,
    24346051,
    24441114,
    24501023,
    24801380,
    25119967,
    26216343,
    26397336,
    26430386,
    26472720,
    27369397,
    27436439,
    27794581,
    28248773,
    28805086,
    29385909,
    29431612,
    30398623,
    31034079,
    32090486,
    32159626,
    32299227,
    32437494,
    33675734,
    33995822,
    34150443,
    35055827,
    36122226,
    36224615,
    36248579,
    36310501,
    36411804,
    36714942,
    36820331,
    36917096,
    37073431,
    37274855,
    37366353,
    38017819,
    38215549,
    38515846,
    38637136,
    38754789,
    39156725,
    39577918,
    39999801,
    40321028,
    42055117,
    42073622,
    42073634,
    42492390,
    42913412,
    47353465,
    47676399,
    48431834,
    48345937,
    49225651,
    49283391,
    49790315,
    50088840,
    50116568,
    50219017,
    50582061,
    50692996,
    51383365,
    52849276,
    53380728,
    54537202,
    55617139,
    55974901,
    56623745,
    57229979,
    57740146,
    58332426,
    58620703,
    60047018,
    60783536,
    62394004,
    63547896,
    63709749,
    63772764,
    63879105,
    64804114,
    65346741,
    66174583,
    66370390,
    67211306,
    67392010,
    69399420,
    71010695,
    71825331,
    72358993,
    72534268,
    72797537,
    72843090,
    73372039,
    73629599,
    74855770,
    77171621,
    80278934,
    81999208,
    85888853,
    92072987,
    93245355,
    94882542,
    94882544,
    97055774,
    100202105,
    103644009,
    115947461,
    120686145,
    121628123,
    122238406,
    123669333,
    123980449,
    126489106,
    129969362,
    132376068,
    132476683,
    132476691,
    133713565,
    134161034,
    135432538,
    136290899,
    138580332,
    139913365,
    140931210,
    144749443,
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
    return "", 0, 0


@cache
def lookup_location(name: str) -> tuple[float, float]:
    """Tries to get the coordinates for a location name.

    Args:
        name (str): The name of the location.

    Returns:
        tuple[float, float]: The latitude and longitude.
    """
    try:
        with open(GEOPY_CACHE, encoding="utf-8") as f:
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


def convert_equipment(equipment: list[dict[str, str | int]]) -> tuple[str, ...]:
    """Converts the equipment to a list of strings.

    Also filters the data by removing everything that is not a clothing item and applying the
    specified conversions
    """
    # apply replacements and remove everything that is not a clothing item
    out = [CLOTHING_REPLACEMENTS.get(name, name) for d in equipment
           if (name := d.get("name")) in ALL_CLOTHING_ITEMS or name in INVALID_CLOTHING
           # also keep items if they might be reduced to a single item later
           or name in [e for t in CLOTHING_REPLACEMENTS_MULTI for e in t]]
    out = replace_tuple(out, CLOTHING_REPLACEMENTS_MULTI)
    return tuple(out)  # convert to tuple to make it hashable


def replace_tuple(data: list[str], mapping: dict[tuple[str, ...], tuple[str, ...]]) -> list[str]:
    """If every element of key tuple is in data, replace it with the value.

    The value is inserted once and the list is modified in place
    """
    for key, val in mapping.items():
        if all(k in data for k in key):
            for k in key:
                data.remove(k)
            data.extend(val)

    return data


def split_equipment(equipment: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """Splits the equipment into categories.

    Args:
        equipment (list[str]): The equipment as list of strings.

    Returns:
        dict[str, list[str]]: The equipment split into categories. The keys are the categories and
                              the values are the items in the category.
    """
    return {k: tuple(e for e in equipment if e in v) for k, v in SORTED_CLOTHING.items()}


def valid_outfits(candidate_outfits: list[dict[str, int]], is_race: bool) -> list[dict[str, int]]:
    """Takes a list of candidate outfits and filters the list for valid outfits.

    Currently this means that the necessary race clothing must be present in the outfit if this is a
    race and race only clothing not be present if it is not a race.

    Args:
        candidate_outfits (list[dict[str, int]]): The list of candidate outfits with the keys being
                                                  the layers and the values being the int encoded
                                                  clothing items
        is_race (bool): Whether this is a race or not

    Returns:
        list[dict[str, int]]: The list of valid outfits, same format as the input
    """
    return [outfit for outfit in candidate_outfits
                      if (not is_race
                          # if not a race, the outfit must not have race only clothing
                          and not any(any(SORTED_CLOTHING[category][v-1] in CLOTHING_ONLY_FOR_RACES
                                          for k, v in outfit.items() if k.startswith(category))
                                      for category in SORTED_CLOTHING))
                      # if a race, the outfit must have some of the necessary race clothing
                      or (is_race and all(any(SORTED_CLOTHING[category][v-1] in items
                                              for k, v in outfit.items() if k.startswith(category))
                                          for category, items in NECESSARY_RACE_CLOTHING.items()))]
