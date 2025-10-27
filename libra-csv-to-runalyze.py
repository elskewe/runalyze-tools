"""Reads a CSV file from Libra and uploads new weight values to Runalyze."""

import json

import click
from rich.progress import MofNCompleteColumn, Progress, TimeElapsedColumn

from runalyze import api


@click.command()
@click.argument("csvfile", type=click.File())
def main(csvfile):
    """Reads a CSV file from Libra and uploads new weight values to Runalyze."""
    data_runalyze = load_weight_from_runalyze()


def load_weight_from_runalyze():
    """Loads the data from Runalyze.

    Caching is unfortunately not really possible as the data is always returned starting from the
    least recent data points."""

    with open("runalyze_credentials.json", encoding="utf-8") as f:
        credentials = json.load(f)

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


if __name__ == "__main__":
    main()
