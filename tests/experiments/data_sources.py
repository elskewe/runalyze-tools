import json
from pathlib import Path

with open(Path(__file__).parent / "dataSources.json", "r", encoding="utf-8") as f:
    data_sources = json.load(f)["dataSource"]

print("hello")
