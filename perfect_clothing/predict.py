import json
import pickle
import re
import pandas as pd
from rich.table import Table
from rich.console import Console
from perfect_clothing import assumptions


def predict(input_data: dict):
    """Predicts the best outfit for the given input data.

    Args:
        input_data (dict): The input data with the keys specified in assumptions.INPUT_COLUMNS
    """
    with open(assumptions.MODEL_FILENAME, "rb") as f:
        model = pickle.load(f)
    with open(assumptions.CANDIDATE_OUTFITS_FILENAME, "r", encoding="utf-8") as f:
        candidate_outfits = json.load(f)
    best_outfits = recommend_best(pd.DataFrame(input_data, index=[1]), model, candidate_outfits, top_k=5)
    table = Table("Rank", "Outfit", "P(ok) in %", title="Best Outfits")
    for i, (p_ok, p_zuHeiss, outfit) in enumerate(best_outfits):
        table.add_row(f"{i+1}.", outfit_to_string(outfit), str(int(p_ok*100)))

    console = Console()
    console.print(table)


def score_outfit(input_data, outfit_encoding, model):
    x = pd.concat((input_data[assumptions.INPUT_COLUMNS], pd.DataFrame(outfit_encoding, index=input_data.index)), axis=1)
    probs = model.predict_proba(x)[0]
    return probs[list(assumptions.TEMPERATURE_LABEL_MAPPING).index("ok")]  # P(ok | weather, outfit)


def recommend_best(input_data, model, candidate_outfits, top_k=1):
    scored = []
    for outfit in candidate_outfits:
        p_ok = score_outfit(input_data, outfit, model)
        scored.append((p_ok, outfit))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:top_k]   # best outfit(s) and their P(ok)


def outfit_to_string(outfit_encoding: dict[str, int]):
    clothing = []
    for category, item in outfit_encoding.items():
        if item > 0:
            category = re.sub(r"_layer\d", "", category)
            clothing.append(assumptions.SORTED_CLOTHING[category][item-1])

    return ", ".join(clothing)
