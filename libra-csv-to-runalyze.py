"""Reads a CSV file from Libra and uploads new weight values to Runalyze."""

import click


@click.command()
@click.argument("csvfile", type=click.File())
def main(csvfile):
    """Reads a CSV file from Libra and uploads new weight values to Runalyze."""


if __name__ == "__main__":
    main()
