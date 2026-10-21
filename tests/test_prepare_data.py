import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

import pandas as pd
import pytest

from perfect_clothing import assumptions
from perfect_clothing.prepare_data import clean_data


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
