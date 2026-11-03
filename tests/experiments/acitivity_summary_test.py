import importlib
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
gfr = importlib.import_module("google-fit-to-runalyze")

with open(Path(__file__).parent / "health_exercise_example.json", "r", encoding="utf-8") as f:
    health_exercises = json.load(f)

walking_data = gfr.extract_activity_data(
    health_exercises["dataPoints"], gfr.WALKING_ACTIVITY_TYPES
)
print(walking_data)
