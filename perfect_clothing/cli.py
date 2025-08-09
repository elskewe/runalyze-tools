import click


@click.group()
def cli():
    pass


@cli.command()
def load_data():
    """Loads the data from Runalyze for training."""
    raise NotImplementedError


@cli.command()
def train():
    """Trains the model."""
    raise NotImplementedError


@cli.command()
def evaluate():
    """Returns the perfect clothing"""
    raise NotImplementedError
