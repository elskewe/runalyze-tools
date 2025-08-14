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
    table = Table("Rank", "Outfit", "P(ok) in %", "P(zuHeiss) in %", "Sum in %", title="Best Outfits")
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


def score_outfit(input_data: pd.DataFrame, outfit_encoding, model) -> tuple[np.float64, np.float64]:
    """Scores the given outfit for the given input data and returns the P(ok) and P(zuHeiss)."""
    x = pd.concat((input_data[assumptions.INPUT_COLUMNS], pd.DataFrame(outfit_encoding, index=input_data.index)), axis=1)
    probs = model.predict_proba(x)[0]
    labels = list(assumptions.TEMPERATURE_LABEL_MAPPING)
    return probs[labels.index("ok")], probs[labels.index("zuHeiss")]


def recommend_best(input_data: pd.DataFrame, model, candidate_outfits, top_k=1) -> pd.DataFrame:
    scored = []
    for outfit in candidate_outfits:
        if input_data["is_race"].item() and \
            not all(any(assumptions.SORTED_CLOTHING[category][v-1] in items
                        for k, v in outfit.items() if k.startswith(category))
                        for category, items in assumptions.NECESSARY_RACE_CLOTHING.items()):
            continue  # outfit without a necessary race clothing item
        p_ok, p_zuHeiss = score_outfit(input_data, outfit, model)
        scored.append((outfit, p_ok, p_zuHeiss))
    scored.sort(reverse=True, key=lambda x: x[1] + x[2])

    # convert to dataframe
    df = pd.DataFrame(scored, columns=["outfit", "P(ok)", "P(zuHeiss)"])
    df["outfit"] = df["outfit"].apply(outfit_to_string)
    # convert to percent
    df["P(ok)"] = df["P(ok)"].transform(lambda x: int(x*100))
    df["P(zuHeiss)"] = df["P(zuHeiss)"].transform(lambda x: int(x*100))
    df["sum"] = df["P(ok)"] + df["P(zuHeiss)"]
    return df[:top_k]   # best outfit(s) and their P(ok) and P(zuHeiss)


def outfit_to_string(outfit_encoding: dict[str, int]):
    clothing = []
    for category, item in outfit_encoding.items():
        if item > 0:
            category = re.sub(r"_layer\d", "", category)
            clothing.append(assumptions.SORTED_CLOTHING[category][item-1])

    return ", ".join(clothing)
