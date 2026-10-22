import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

import pandas as pd
import pytest

from perfect_clothing import assumptions
from perfect_clothing.prepare_data import clean_data, save_candidate_outfits


@pytest.mark.parametrize(
    "recency_years, expected_ids",
    [
        (3, [1]),
        (1, []),
    ],
)
def test_clean_data_keeps_recent_infrequent_outfits(monkeypatch, recency_years, expected_ids):
    now = pd.Timestamp.now(tz="UTC")
    data = pd.DataFrame(
        {
            "equipment": [("outfit 1",), ("outfit 2",)],
            "date_time": [
                now - pd.DateOffset(years=2),
                now - pd.DateOffset(years=4),
            ],
            "comfort": ["ok", "ok"],
            "upper_body_layer1": [1, 1],
            "lower_body_layer1": [1, 1],
        },
        index=[1, 2],
    )
    monkeypatch.setattr(assumptions, "OUTFIT_FREQUENCY_THRESHOLD", 2)
    monkeypatch.setattr(assumptions, "OUTFIT_FREQUENCY_THRESHOLD_EXCEPTIONS", [])
    monkeypatch.setattr(assumptions, "OUTFIT_RECENCY_THRESHOLD_YEARS", recency_years)
    monkeypatch.setattr(assumptions, "valid_outfits", lambda outfits, is_race: [])

    cleaned, _ = clean_data(data, ["upper_body_layer1", "lower_body_layer1"])

    assert cleaned.index.tolist() == expected_ids


def test_save_candidate_outfits_uses_shared_frequency_exceptions(monkeypatch, tmp_path):
    now = pd.Timestamp.now(tz="UTC")
    encoded_columns = ["upper_body_layer1", "lower_body_layer1", "legs_layer1"]
    data = pd.DataFrame(
        {
            "equipment": [
                ("Kompressionsstrümpfe",),
                ("Kompressionsstrümpfe",),
                (),
            ],
            "date_time": [
                now - pd.DateOffset(years=2),
                now - pd.DateOffset(years=4),
                now - pd.DateOffset(years=4),
            ],
            "upper_body_layer1": [1, 2, 3],
            "lower_body_layer1": [1, 1, 1],
            "legs_layer1": [0, 0, 1],
        }
    )
    monkeypatch.setattr(assumptions, "OUTFIT_FREQUENCY_THRESHOLD", 2)
    monkeypatch.setattr(assumptions, "OUTFIT_FREQUENCY_THRESHOLD_EXCEPTIONS", ["Kompressionsstrümpfe"])
    monkeypatch.setattr(assumptions, "OUTFIT_RECENCY_THRESHOLD_YEARS", 3)
    monkeypatch.setattr(assumptions, "CANDIDATE_OUTFITS_FILENAME", str(tmp_path / "outfits.json"))

    save_candidate_outfits(data, encoded_columns)
    with open(assumptions.CANDIDATE_OUTFITS_FILENAME, encoding="utf-8") as f:
        outfits = json.load(f)

    assert [outfit["race_only"] for outfit in outfits] == [False, True, False]
    assert [outfit["n_worn"] for outfit in outfits] == [1, 1, 1]
