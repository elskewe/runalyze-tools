"""Reads a CSV file from Libra and uploads new weight values to Runalyze."""

import datetime
import json
import math

import click
import pandas as pd
from rich.progress import MofNCompleteColumn, Progress, TimeElapsedColumn

from runalyze import api


@click.command()
@click.argument("csvfile", type=click.File())
def main(csvfile):
    """Reads a CSV file from Libra and uploads new weight values to Runalyze."""
    with open("runalyze_credentials.json", encoding="utf-8") as f:
        credentials = json.load(f)

    data_runalyze = load_weight_from_runalyze(credentials)
    data_libra = load_libra_csv(csvfile)
    missing_data = find_missing_data(data_runalyze, data_libra)


def load_weight_from_runalyze(credentials):
    """Loads the data from Runalyze.

    Caching is unfortunately not really possible as the data is always returned starting from the
    least recent data points."""

    page = 1
    data = []

    with Progress(*Progress.get_default_columns(), MofNCompleteColumn(), TimeElapsedColumn()) as p:
        task_id = p.add_task("Loading data", total=None)
        while True:
            new_data = api.get_body_composition(credentials, page)
            data.extend(new_data)
            page += 1
            p.update(task_id, advance=1)
            if not new_data:
                break  # stop when there is no new data

    print(f"Loaded {len(data)} data points")
    return data

def load_libra_csv(file) -> pd.DataFrame:
    """Reads a CSV file from Libra."""
    return pd.read_csv(file, sep=";", skiprows=3, parse_dates=[0])

def find_missing_data(data_runalyze: list, data_libra: pd.DataFrame, abs_tol=0.15):
    """Returns a dataframe which contains the libra data which is not found on Runalyze."""

    # construct dict for more efficient lookup
    data_runalyze_dict: dict[datetime.date, float] = {pd.to_datetime(d["date_time"]).date(): d["weight"] for d in data_runalyze}

    return data_libra[data_libra[["#date", "weight"]].apply(
        lambda r: not (data_runalyze_dict.get(libra_date :=r["#date"].date())
                       # sometimes the date in Runalyze is one day later or earlier. Only match
                       # these if the weight is (mostly) the same
                       or any((weight := data_runalyze_dict.get(libra_date + delta))
                              and math.isclose(r["weight"], weight, abs_tol=abs_tol)
                              for delta in [-datetime.timedelta(days=1), datetime.timedelta(days=1)])),
                      axis=1
    )]


if __name__ == "__main__":
    main()
