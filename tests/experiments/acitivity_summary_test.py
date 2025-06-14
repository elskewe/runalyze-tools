import importlib
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
gfr = importlib.import_module("google-fit-to-runalyze")

with open(Path(__file__).parent / "activity_summary_example.json", "r") as f:
    activity_summary = json.load(f)

d = gfr.extract_data_from_aggregated_data(activity_summary)
d_ = gfr.extract_acitivity_segment_data(d, [7, 8])

print(d)
