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
    pr.predict({"wind_chill": -5, "x_gap": 12, "ghi": 300, "is_race": False})
