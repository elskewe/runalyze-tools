import json
import pickle
import re
import numpy as np
import pandas as pd
from rich.table import Table
from rich.console import Console
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
    with open(candidate_outfits_filename, "r", encoding="utf-8") as f:
        candidate_outfits = json.load(f)
    return model, candidate_outfits


def score_outfits(input_data: pd.DataFrame, outfit_encoding: list[dict[str, int]], model) -> np.ndarray:
    """Scores the given outfit for the given input data and returns the P(ok) and P(zuHeiss)."""
    # repeats the relevant columns of `input_data` `len(outfit_encoding)` times and then merges the
    # outfits s.t. the result has the same number of rows as there are entries (i.e. outfits) in
    # `outfit_encoding`
    x = pd.concat([input_data[assumptions.INPUT_COLUMNS]]*len(outfit_encoding), ignore_index=True) \
        .join(pd.DataFrame(outfit_encoding))
    probs = model.predict_proba(x)
    return probs


def recommend_best(input_data: pd.DataFrame, model, candidate_outfits, top_k=1) -> pd.DataFrame:
    labels = list(assumptions.TEMPERATURE_LABEL_MAPPING)
    # only score outfit with a necessary race clothing item if it is a race
    valid_outfits = [outfit for outfit in candidate_outfits
                     if not input_data["is_race"].item() or
                     all(any(assumptions.SORTED_CLOTHING[category][v-1] in items
                             for k, v in outfit.items() if k.startswith(category))
                         for category, items in assumptions.NECESSARY_RACE_CLOTHING.items())]
    probs = score_outfits(input_data, valid_outfits, model)
    scored = [(outfit, p[labels.index("ok")], p[labels.index("zuHeiss")],
               p[labels.index("zuKaltAngezogen")], p[labels.index("zuWarmAngezogen")])
              for outfit, p in zip(valid_outfits, probs)]
    scored.sort(reverse=True, key=lambda x: x[1] + x[2])

    # convert to dataframe
    df = pd.DataFrame(scored, columns=["outfit", "P(ok)", "P(zuHeiss)", "P(zuKalt)", "P(zuWarm)"])
    df["outfit"] = df["outfit"].apply(outfit_to_string)
    # convert to percent
    for col in ["P(ok)", "P(zuHeiss)", "P(zuKalt)", "P(zuWarm)"]:
        df[col] = df[col].transform(lambda x: int(x*100))
    df["sum_ok"] = df["P(ok)"] + df["P(zuHeiss)"]
    return df[:top_k]   # best outfit(s) and their P(ok) and P(zuHeiss)


def outfit_to_string(outfit_encoding: dict[str, int]):
    clothing = []
    for category, item in outfit_encoding.items():
        if item > 0:
            category = re.sub(r"_layer\d", "", category)
            clothing.append(assumptions.SORTED_CLOTHING[category][item-1])

    return ", ".join(clothing)
