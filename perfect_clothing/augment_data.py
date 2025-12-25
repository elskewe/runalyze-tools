"""Augments the data by adding synthetic samples based on domain knowledge."""

from operator import itemgetter

import numpy as np
import pandas as pd

from perfect_clothing import assumptions, weather
from perfect_clothing.prepare_data import get_radiation_data


def augment_data(data: pd.DataFrame, encoded_clothing_columns: list[str],
                 candidate_outfits: list[dict[str, int]]) -> pd.DataFrame:
    """Adds more examples where the label is not ok.

    Basic idea: if an outfit is too warm, every outfit that only contains the same or warmer items
    will also be too warm. The same principle holds for the other labels.

    Furthermore, to balance the categories afterwards, the other features might be modified as well
    to get more data for less frequent categories.
    """
    new_rows = []

    for id_, current_row in data.iterrows():
        new_rows.extend(generate_new_outfit(id_, current_row, encoded_clothing_columns, candidate_outfits))

    for factor in range(1, assumptions.MAX_AUGMENTATION_FACTOR+1):
        # get comfort labels so far
        comfort_int = pd.concat([data["comfort_int"], pd.Series([e["comfort_int"] for e in new_rows])])
        # the return of `value_counts` is sorted by frequency, thus the first value of the index is
        # always the most frequent label
        most_frequent_comfort_label = int(comfort_int.value_counts().index[0])  # type: ignore
        least_frequent_comfort_label = comfort_int.value_counts().index[-1]
        if least_frequent_comfort_label == assumptions.TEMPERATURE_LABEL_MAPPING[assumptions.OK_TEMPERATURE_LABEL]:
            break  # continue this loop until `ok` is the least frequent label (it can't be augmented)
        for id_, current_row in data.iterrows():
            new_rows.extend(generate_new_features(id_, current_row, most_frequent_comfort_label, factor))

    for id_, current_row in data.iterrows():
        new_rows.extend(generate_new_features_ok(id_, current_row, encoded_clothing_columns, assumptions.MAX_AUGMENTATION_FACTOR))

    # add to base dataframe
    return pd.concat([data, pd.DataFrame.from_records(new_rows, index="id")])


def generate_new_outfit(id_: int, base_row: pd.Series, encoded_clothing_columns: list[str],
                        candidate_outfits: list[dict[str, int]]) -> list[dict]:
    """Generates new outfits based on the given label (`base_row["comfort_int"]`) and note.

    The new outfits are warmer/colder (depending on the label) in at least one clothing category and
    at least the same in the rest, which means the direction of the comfort (too cold/too warm) is
    retained. The new outfit (with the other data from the base_row) is than added to the return
    value. Besides the actual label, an attempt is made to determine the sentiment of the note. This
    sentiment is used like the actual label if they do not contradict each other. The idea here is
    the same, if an outfit is almost too warm, then adding items will finally push that outfit into
    being too warm (the same principle applies for too cold).

    Args:
        base_row (pd.Series): the row from which the new outfits are generated
        encoded_clothing_columns (list[str]): the names of the clothing columns candidate_outfits
        (list[dict[str, int]]): the possible outfits

    Returns:
        list[dict]: the additional rows with the altered outfits
    """
    outfit = base_row[encoded_clothing_columns].to_dict()
    # The sentiment of the note is "zuWarmAngezogen" (`1`) if it contains any word from
    # `ALMOST_TOO_WARM_WORDS` and "zuKaltAngezogen" (`-1`) if it contains any word from
    # `ALMOST_TOO_COLD_WORDS`. If words from both list or none are present, the note is not used as
    # it is ambiguous.
    note_sentiment = (any(w in base_row["note"] for w in assumptions.ALMOST_TOO_WARM_WORDS)
        - any(w in base_row["note"] for w in assumptions.ALMOST_TOO_COLD_WORDS)) \
            if isinstance(base_row["note"], str) else 0
    label = base_row["comfort_int"]
    # If the inferred label from the note and the actual label contradict each other disregard the
    # note.
    if label * note_sentiment < 0:
        note_sentiment = 0

    if label > 0 or note_sentiment > 0:  # zuWarmAngezogen
        # Generate outfits warmer compared to this one
        new_label = "zuWarmAngezogen"
        new_rows = [candidate_outfit for candidate_outfit in candidate_outfits
                    if all(v >= outfit[k] for k, v in candidate_outfit.items()) and outfit != candidate_outfit
                        and sum(candidate_outfit.values()) - sum(outfit.values()) in assumptions.MAX_CLOTHING_DISTANCE]
    elif label < 0 or note_sentiment < 0:  # zuKaltAngezogen
        # Generate outfits cooler compared to this one
        new_label = "zuKaltAngezogen"
        new_rows = [candidate_outfit for candidate_outfit in candidate_outfits
                    if all(v <= outfit[k] for k, v in candidate_outfit.items()) and outfit != candidate_outfit
                    and sum(candidate_outfit.values()) - sum(outfit.values()) in assumptions.MAX_CLOTHING_DISTANCE]
    else:  # label == 0
        return []  # no new rows are added

    return [base_row.to_dict() | new_row
            | {"comfort": new_label, "comfort_int": assumptions.TEMPERATURE_LABEL_MAPPING[new_label],
               "id": id_}
            for new_row in new_rows]


def generate_new_features(id_: int, base_row: pd.Series, most_frequent_comfort_label: int,
                          augmentation_factor: int) -> list[dict]:
    """Moves each of the features (by `augmentation_factor`) to get additional rows with the same label.

    To combat label imbalance, this is only done for labels which are not the most frequent one."""
    # Return if the label is `ok` as it's not possible to know which features to move to still
    # retain the same label. Also don't make the imbalance worse by adding new rows of the already
    # most frequent label.
    if base_row["comfort_int"] == assumptions.TEMPERATURE_LABEL_MAPPING[assumptions.OK_TEMPERATURE_LABEL] \
            or base_row["comfort_int"] == most_frequent_comfort_label:
        return []

    new_rows = []
    # amplitude and direction in which the features are moved
    factor = augmentation_factor * np.sign(base_row["comfort_int"])
    for feature, direction in assumptions.INPUT_COLUMNS.items():
        if direction == 0:
            continue
        match feature:
            case "wind_speed" | "temperature":  # also need to adapt windchill
                if feature == "wind_speed":
                    new_wind_speed = base_row["wind_speed"] + direction * factor
                    new_temperature = base_row["temperature"]
                else:  # feature == "temperature":
                    new_wind_speed = base_row["wind_speed"]
                    new_temperature = base_row["temperature"] + direction * factor
                if new_wind_speed < 0:
                    continue  # invalid wind speed
                new_rows.append({"wind_speed": new_wind_speed,
                                 "temperature": new_temperature,
                                 "wind_chill": weather.wind_chill(new_temperature, new_wind_speed)})
            case "ghi_start":
                # I don't like the hardcoded value here, but it's the easiest way right now
                new_cloud_cover_perc = base_row["cloud_cover"] - 50 * factor
                if new_cloud_cover_perc < 0 or new_cloud_cover_perc > 100:
                    continue  # cloud cover is already at minimum or maximum
                tmp_df = pd.DataFrame([base_row])
                tmp_df["cloud_cover"] = new_cloud_cover_perc
                # the `group` is only to get the correct type (dataframe instead of series)
                tmp_df[["ghi_start", "ghi_middle", "ghi_end"]] = \
                    tmp_df.groupby(["latitude", "longitude"], group_keys=False).apply(get_radiation_data, include_groups=False)
                new_rows.append({"cloud_cover": new_cloud_cover_perc,
                                 "ghi_start": tmp_df["ghi_start"].iloc[0],
                                 "ghi_middle": tmp_df["ghi_middle"].iloc[0],
                                 "ghi_end": tmp_df["ghi_end"].iloc[0]})
            case "ghi_middle" | "ghi_end":
                continue  # already handled by `ghi_start`
            case "x_pace":
                new_x_pace = base_row["x_pace"] + direction * factor
                if new_x_pace < 0:
                    continue  # invalid x_pace
                new_rows.append({feature: new_x_pace})
            case _:
                new_rows.append({feature: base_row[feature] + direction * factor})

    return [base_row.to_dict() | new_row | {"id": id_} for new_row in new_rows]


def generate_new_features_ok(id_: int, base_row: pd.Series, encoded_clothing_columns: list[str], augmentation_factor: int) -> list[dict]:
    """Moves the temperature (by `augmentation_factor`) to get additional rows with the label changed correspondingly.

    Only looks at rows where the temperature label is ok. To get reasonable results, the augmentation factor should be
    high enough that the new features definitely change the label."""

    if base_row["comfort_int"] != assumptions.TEMPERATURE_LABEL_MAPPING[assumptions.OK_TEMPERATURE_LABEL]:
        return []

    new_rows = []
    # amplitude and direction in which the features are moved
    for new_comfort_int in itemgetter("zuKaltAngezogen", "zuWarmAngezogen")(assumptions.TEMPERATURE_LABEL_MAPPING):
        factor = augmentation_factor * new_comfort_int
        for feature, direction in assumptions.INPUT_COLUMNS.items():
            if direction == 0:
                continue
            match feature:
                case "temperature":  # also need to adapt windchill
                    new_temperature = base_row["temperature"] + direction * factor
                    new_comfort_int_ = assumptions.TEMPERATURE_LABEL_MAPPING["zuHeiss"] \
                        if (new_comfort_int > assumptions.TEMPERATURE_LABEL_MAPPING["ok"]
                            and assumptions.can_outfit_be_toHeiss(base_row[encoded_clothing_columns].to_dict(), base_row["is_race"])) \
                        else new_comfort_int
                    new_rows.append({"temperature": new_temperature,
                                    "wind_chill": weather.wind_chill(new_temperature, base_row["wind_speed"]),
                                    "comfort_int": new_comfort_int_})
                case _:  # it's not certain enough that the new features change the label
                    continue

    return [base_row.to_dict() | new_row | {"id": id_} for new_row in new_rows]
