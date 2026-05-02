"""Augments the data by adding synthetic samples based on domain knowledge."""

from datetime import timedelta
from operator import itemgetter
from typing import cast

import numpy as np
import pandas as pd
from rich.progress import track, Progress

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

    for id_, current_row in track(data.iterrows(), total=len(data),
                                  description="Augmenting data with new outfits and durations"):
        new_rows.extend(generate_new_outfit(id_, current_row, encoded_clothing_columns, candidate_outfits))
        new_rows.extend(generate_new_duration(id_, current_row))

    with Progress() as progress:
        augmentation_range = range(1, assumptions.MAX_AUGMENTATION_FACTOR+1)
        task = progress.add_task(description="Augmenting data with new features")
        for i, factor in enumerate(augmentation_range):
            # get comfort labels so far
            comfort_int = pd.concat([data["comfort_int"], pd.Series([e["comfort_int"] for e in new_rows])])
            # the return of `value_counts` is sorted by frequency, thus the first value of the index is
            # always the most frequent label
            most_frequent_comfort_label = int(comfort_int.value_counts().index[0])  # type: ignore
            least_frequent_comfort_label = comfort_int.value_counts().index[-1]
            if least_frequent_comfort_label == assumptions.TEMPERATURE_LABEL_MAPPING[assumptions.OK_TEMPERATURE_LABEL]:
                break  # continue this loop until `ok` is the least frequent label (it can't be augmented)
            for id_, current_row in progress.track(data.iterrows(), task_id=task,
                                                   total=len(data)*len(augmentation_range), completed=i*len(data)):
                new_rows.extend(generate_new_features(id_, current_row, most_frequent_comfort_label,
                                                      encoded_clothing_columns, factor))

    for id_, current_row in track(data.iterrows(), total=len(data),
                                  description="Augmenting data with new features which change the label"):
        new_rows.extend(generate_new_features_ok(id_, current_row, encoded_clothing_columns, assumptions.MAX_AUGMENTATION_FACTOR))

    new_rows.extend(merge_activities(data, encoded_clothing_columns))

    # add to base dataframe
    new_rows = [r | {"is_augmented": True} for r in new_rows]
    return pd.concat([data, pd.DataFrame.from_records(new_rows, index="id")])


def generate_new_outfit(id_: int, base_row: pd.Series, encoded_clothing_columns: list[str],
                        candidate_outfits: list[dict[str, int]]) -> list[dict]:
    """Generates new outfits based on the given label (`base_row["comfort_int"]`) and note.

    The new outfits are warmer/colder (depending on the label) in at least one clothing category and
    at least the same in the rest, which means the direction of the comfort (too cold/too warm) is
    retained. The new outfit (with the other data from the base_row) is than added to the return
    value. Besides the actual label, the sentiment of the note is used. The idea here is the same,
    if an outfit is almost too warm, then adding any item will finally push that outfit into being
    too warm (the same principle applies for too cold).

    Args:
        base_row (pd.Series): the row from which the new outfits are generated
        encoded_clothing_columns (list[str]): the names of the clothing columns candidate_outfits
        (list[dict[str, int]]): the possible outfits

    Returns:
        list[dict]: the additional rows with the altered outfits
    """
    outfit = base_row[encoded_clothing_columns].to_dict()
    label = base_row["comfort_int"]
    note_sentiment = base_row["note_sentiment"]

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


def generate_new_duration(id_: int, base_row: pd.Series) -> list[dict]:
    """Adds new rows by adjusting the duration slightly

    The assumptions being that this wont change the label
    """
    return [base_row.to_dict() | {"id": id_, "duration": base_row["duration"]*factor}
            for factor in [1 + assumptions.INVARIANT_DURATION_CHANGE, 1 - assumptions.INVARIANT_DURATION_CHANGE]]


def generate_new_features(id_: int, base_row: pd.Series, most_frequent_comfort_label: int,
                          encoded_clothing_columns: list[str], augmentation_factor: int) -> list[dict]:
    """Moves each of the features (by `augmentation_factor`) to get additional rows with the same label.

    To combat label imbalance, this is only done for labels which are not the most frequent one. The
    idea is the same as `generate_new_outfit`, i.e. if an outfit is either already too cold or
    almost too cold, reducing e.g. the temperature will definitely mean that outfit is too cold. The
    same principle holds for too warm. The direction and magnitude each feature has to be moved is
    encoded in the value of `assumptions.INPUT_COLUMNS`.
    """
    label = base_row["comfort_int"]
    note_sentiment = base_row["note_sentiment"]
    # Return if the label is `ok` (and there is no further information in the note) as it's not
    # possible to know which features to move to still retain the same label. If only the
    # information in the note is present (but the label is `ok`) return if the augementation factor
    # is 1 or less as with that factor it is not certain enough that the label actually changes.
    # Also don't make the imbalance worse by adding new rows of the already most frequent label.
    if (label == assumptions.TEMPERATURE_LABEL_MAPPING[assumptions.OK_TEMPERATURE_LABEL]
        and (note_sentiment == label or augmentation_factor <= 1))\
            or label == most_frequent_comfort_label:
        return []

    new_label_direction = np.sign(label + note_sentiment).item()
    new_label = _new_label(base_row, encoded_clothing_columns, new_label_direction)
    new_rows = []
    # amplitude and direction in which the features are moved
    factor = augmentation_factor * new_label_direction

    # helper for the GHI threshold check. Returns True iff the augmentation should be kept.
    def _ghi_threshold_ok(base_row: pd.Series, tmp_df: pd.DataFrame) -> bool:
        if label == 0 and note_sentiment != 0:
            base_avg = base_row[["ghi_start", "ghi_middle", "ghi_end"]].mean()
            new_avg = tmp_df[["ghi_start", "ghi_middle", "ghi_end"]].iloc[0].mean()
            return abs(new_avg - base_avg) > assumptions.GHI_AUGMENTATION_THRESHOLD
        return True

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
                if not _ghi_threshold_ok(base_row, tmp_df):
                    continue
                new_rows.append({"cloud_cover": new_cloud_cover_perc,
                                 "ghi_start": tmp_df["ghi_start"].iloc[0],
                                 "ghi_middle": tmp_df["ghi_middle"].iloc[0],
                                 "ghi_end": tmp_df["ghi_end"].iloc[0]})
            case "ghi_middle":
                continue  # already handled by `ghi_start`
            case "ghi_end":
                # This is not quite the right place for changing the time (it might as well be for
                # `ghi_start`), but the implementation is relatively straightforward this way.
                if base_row["ghi_start"] > base_row["ghi_middle"] and base_row["ghi_middle"] >= base_row["ghi_end"]:
                    # after noon (the latter comparison can be equal when the sun sets during the activity)
                    direction = -1
                elif base_row["ghi_start"] <= base_row["ghi_middle"] and base_row["ghi_middle"] < base_row["ghi_end"]:
                    # before noon (the first comparison can be equal when the sun rises during the activity)
                    direction = 1
                else:
                    # around noon, no clear direction to move the time
                    continue
                new_date_time = base_row["date_time"] + timedelta(hours=direction * factor)
                tmp_df = pd.DataFrame([base_row])
                tmp_df["date_time"] = new_date_time
                ghi_columns = ["ghi_start", "ghi_middle", "ghi_end"]
                # The `group` is only to get the correct type (dataframe instead of series). This is
                # rather slow, but I don't see an easy way to improve performance.
                tmp_df[ghi_columns] = tmp_df.groupby(["latitude", "longitude"], group_keys=False) \
                    .apply(get_radiation_data, include_groups=False)

                diff_sign = np.sign((tmp_df[ghi_columns] - base_row[ghi_columns]).iloc[0].to_numpy())
                if (np.sign(diff_sign + factor) != np.sign(factor)).any():
                    # The new radiation values don't all change in the right direction (or none
                    # change at all), thus this augmentation is dropped (this can e.g. occur when
                    # the time is put on the other side of the noon)
                    continue
                if not _ghi_threshold_ok(base_row, tmp_df):
                    continue
                new_rows.append({"date_time": new_date_time,
                                 "ghi_start": tmp_df["ghi_start"].iloc[0],
                                 "ghi_middle": tmp_df["ghi_middle"].iloc[0],
                                 "ghi_end": tmp_df["ghi_end"].iloc[0]})
            case "x_pace":
                new_x_pace = base_row["x_pace"] + direction * factor
                new_pace = new_x_pace / base_row["variability_index_pace"]
                if new_x_pace < 0:
                    continue  # invalid x_pace
                new_rows.append({feature: new_x_pace, "pace": new_pace})
            case "stopped_time":
                new_stopped_time = base_row["stopped_time"] + direction * factor
                if new_stopped_time < 0:
                    continue  # invalid stopped_time
                new_rows.append({feature: new_stopped_time})
            case _:
                new_rows.append({feature: base_row[feature] + direction * factor})

    return [base_row.to_dict() | new_row | {"id": id_, "comfort_int": new_label} for new_row in new_rows]


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
                    new_rows.append({"temperature": new_temperature,
                                    "wind_chill": weather.wind_chill(new_temperature, base_row["wind_speed"]),
                                    "comfort_int": _new_label(base_row, encoded_clothing_columns, new_comfort_int)})
                case _:  # it's not certain enough that the new features change the label
                    continue

    return [base_row.to_dict() | new_row | {"id": id_} for new_row in new_rows]


def merge_activities(data: pd.DataFrame, encoded_clothing_columns: list[str]) -> list[dict]:
    """Merges activities which are close together and have the same clothing and comfort label.

    Also the activity must not be a race
    """
    data = data.copy()  # to avoid modifying the original dataframe
    data["date_time_utc"] = pd.to_datetime(data["date_time"], utc=True)
    data["end_time_utc"] = data["date_time_utc"] + pd.to_timedelta(data["elapsed_time"], unit="s")
    new_rows = []
    id_processed = set()

    for id_, current_row in track(data.iterrows(), total=len(data),
                                  description="Augmenting data with merged activities"):
        if id_ in id_processed or current_row["is_race"]:
            continue

        group_ids = {id_}
        similar_activities = assumptions.find_similar_activities(data, current_row, encoded_clothing_columns,
                                                                 id_processed)
        group_ids.update(similar_activities.index)
        queue = [row for _, row in similar_activities.iterrows()]

        while queue:
            row = queue.pop()
            similar_to_row = assumptions.find_similar_activities(data, row, encoded_clothing_columns,
                                                                 id_processed | group_ids)
            new_ids = set(similar_to_row.index) - group_ids
            if new_ids:
                group_ids.update(new_ids)
                queue.extend([row for _, row in data.loc[list(new_ids)].iterrows()])

        if len(group_ids) > 1:
            merged_row = assumptions.merge_activities(data.loc[sorted(group_ids)], encoded_clothing_columns)
            new_rows.append(merged_row.to_dict() | {"id": merged_row.name, "is_merged": True})

        id_processed.update(group_ids)

    return new_rows


def _new_label(row: pd.Series, encoded_clothing_columns: list[str], new_label: int) -> int:
    """Helper to determine whether the new should be changed to `zuHeiss`
    """
    return assumptions.TEMPERATURE_LABEL_MAPPING["zuHeiss"] \
        if (new_label > assumptions.TEMPERATURE_LABEL_MAPPING["ok"]
            and assumptions.can_outfit_be_toHeiss(row[encoded_clothing_columns].to_dict(), row["is_race"])) \
        else new_label
