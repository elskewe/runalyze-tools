import click
from perfect_clothing import load_data as ld


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
    raise NotImplementedError


@cli.command()
def evaluate():
    """Returns the perfect clothing"""
    raise NotImplementedError
