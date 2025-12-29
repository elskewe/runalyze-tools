import json
import pickle
import re

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from perfect_clothing import assumptions


def predict(input_data: dict):
    """Predicts the best outfit for the given input data.

    Args:
        input_data (dict): The input data with the keys specified in assumptions.INPUT_COLUMNS
    """
    model, candidate_outfits = load_data()
    best_outfits = recommend_best(pd.DataFrame(input_data, index=[1]), model, candidate_outfits, top_k=100)
    table = Table("Rank", *best_outfits.columns.to_list(), title="Best Outfits")
    for i, row in best_outfits.iterrows():
        table.add_row(f"{i+1}.", *(str(e) for e in row.to_list()))

    console = Console()
    console.print(table)


def load_data(model_filename=assumptions.MODEL_FILENAME,
              candidate_outfits_filename=assumptions.CANDIDATE_OUTFITS_FILENAME):
    with open(model_filename, "rb") as f:
        model = pickle.load(f)
    with open(candidate_outfits_filename, encoding="utf-8") as f:
        candidate_outfits = json.load(f)
    return model, candidate_outfits


def score_outfits(input_data: pd.DataFrame, outfit_encoding: list[dict[str, int]], model) -> np.ndarray:
    """Scores the given outfit for the given input data and returns the P(ok) and P(zuHeiss)."""
    # repeats the relevant columns of `input_data` `len(outfit_encoding)` times and then merges the
    # outfits s.t. the result has the same number of rows as there are entries (i.e. outfits) in
    # `outfit_encoding`
    x = pd.concat([input_data[assumptions.INPUT_COLUMNS.keys()]]*len(outfit_encoding), ignore_index=True) \
        .join(pd.DataFrame(outfit_encoding).drop("n_worn", axis=1))
    return model.predict_proba(x)


def recommend_best(input_data: pd.DataFrame, model, candidate_outfits, top_k=1) -> pd.DataFrame:
    labels = list(assumptions.TEMPERATURE_LABEL_MAPPING)
    # only score outfit with a necessary race clothing item if it is a race
    valid_outfits = assumptions.valid_outfits(candidate_outfits, input_data["is_race"].item())
    probs = score_outfits(input_data, valid_outfits, model)
    df = pd.DataFrame([{"outfit": outfit_to_string(outfit), "n": outfit["n_worn"], "P(ok)": p[labels.index("ok")],
                        "P(kalt)": p[labels.index("zuKaltAngezogen")], "P(warm)": p[labels.index("zuWarmAngezogen")],
                        "P(heiß)": p[labels.index("zuHeiss")]}
                      for outfit, p in zip(valid_outfits, probs)])

    df["sum_ok"] = df["P(ok)"] + df["P(heiß)"]
    # delta between too warm and too cold, positive means outfit is on the warmer side
    df["Δ"] = df["P(warm)"] - df["P(kalt)"]
    # This sorts the results descending by `sum_ok - abs(Δ)`, which is the same as sorting ascending
    # by `abs(Δ) - sum_ok` which is easier to implement (as `sum_ok` can stay unchanged). This means
    # that the best result is the one where `sum_ok` is the highest while the outfit is overall
    # balanced
    df = df.sort_values("sum_ok", ignore_index=True, key=df["Δ"].abs().sub)
    # convert to percent
    for col in df.columns:
        if col.startswith("P(") or col in ["sum_ok", "Δ"]:
            df[col] = df[col].transform(lambda x: int(x*100))
    return df[:top_k]   # best outfit(s) and their P(ok) and P(zuHeiss)


def outfit_to_string(outfit_encoding: dict[str, int]):
    clothing = []
    for category, item in outfit_encoding.items():
        if item > 0 and category.startswith(tuple(assumptions.SORTED_CLOTHING.keys())):
            category = re.sub(r"_layer\d", "", category)
            clothing.append(assumptions.SORTED_CLOTHING[category][item-1])

    return ", ".join(clothing)
