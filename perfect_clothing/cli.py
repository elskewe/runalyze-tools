import click
from perfect_clothing import load_data as ld
from perfect_clothing import train as tr
from perfect_clothing import predict as pr


@click.group()
def cli():
    pass


@cli.command()
def load_data():
    """Loads the data from Runalyze for training."""
    ld.load_data()


@cli.command()
def update_data():
    """Updates the existing data from Runalyze with new activities."""
    ld.update_data()


@cli.command()
def clean_data():
    """Cleans the already loaded data.

    This is already done in `load_data`, but this functionality is useful when
    the logic for cleaning is extended.
    """
    ld.clean_file()


@cli.command()
def train():
    """Trains the model."""
    tr.train()


@cli.command()
def predict():
    """Returns the perfect clothing"""
    pr.predict({"temperature": -4, "wind_chill": 10, "x_gap": 12, 
                "ghi_start": 300, "ghi_middle": 350, "ghi_end": 350,
                "wind_speed": 5, "x_pace": 13.1, "is_race": False, "duration": 1800})
