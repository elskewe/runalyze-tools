import sys
from pathlib import Path

# tests live in a sibling directory of the package; ensure the parent of this file is on
# the import path so that `perfect_clothing` can be imported.
sys.path.append(str(Path(__file__).parents[1]))

import pandas as pd
import pytest

from perfect_clothing import assumptions, augment_data


def make_base_row():
    # minimal set of features used by generate_new_features plus a few extras
    return pd.Series({
        "ghi_start": 100,
        "ghi_middle": 100,
        "ghi_end": 100,
        "cloud_cover": 50,
        "latitude": 0,
        "longitude": 0,
        "date_time": pd.Timestamp("2020-01-01 12:00"),
        "elapsed_time": 3600,
        "comfort_int": 0,
        "note_sentiment": 1,
        "wind_speed": 0,
        "temperature": 20,
        "variability_index_pace": 1,
        "x_pace": 0,
        "stopped_time": 0,
        "pace": 0,
        "is_race": False,
    })


def _run_with_delta(monkeypatch, delta, threshold):
    """Helper: monkeypatches radiation to add `delta` to each GHI component and runs
    `generate_new_features`. Returns the produced rows.
    """
    orig_inputs = assumptions.INPUT_COLUMNS.copy()
    assumptions.INPUT_COLUMNS = {"ghi_start": 1, "ghi_middle": 1, "ghi_end": 1}

    def fake_rad(df, include_groups=False):
        return pd.DataFrame({
            "ghi_start": df["ghi_start"] + delta,
            "ghi_middle": df["ghi_middle"] + delta,
            "ghi_end": df["ghi_end"] + delta,
        }, index=df.index)

    monkeypatch.setattr("perfect_clothing.augment_data.get_radiation_data", fake_rad)
    assumptions.GHI_AUGMENTATION_THRESHOLD = threshold

    rows = augment_data.generate_new_features(
        id_=0,
        base_row=make_base_row(),
        most_frequent_comfort_label=1,
        augmentation_factor=1,
    )

    assumptions.INPUT_COLUMNS = orig_inputs
    return rows


@pytest.mark.parametrize("delta,threshold,expect_any", [
    (1, 5, False),  # small change -> suppressed
    (10, 5, True),  # larger change -> kept
])
def test_ghi_augmentation_delta(monkeypatch, delta, threshold, expect_any):
    rows = _run_with_delta(monkeypatch, delta, threshold)
    if expect_any:
        assert rows, "augmentation should occur when avg change > threshold"
    else:
        assert rows == [], "augmentation should be suppressed when avg change <= threshold"
