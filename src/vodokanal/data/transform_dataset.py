import sys

import click
import pandas as pd

from src.vodokanal.exceptions import CustomException


@click.command()
@click.option(
    '--input_data_path',
    required=True,
    type=click.Path(exists=True),
    prompt='Specify input path',
    help='Path to input data file',
)
@click.option(
    '--output_data_path',
    required=True,
    type=click.Path(),
    prompt='Specify output path',
    help='Path to save output data file',
)
def data_transformation(input_data_path, output_data_path):
    try:
        df = pd.read_csv(input_data_path)
        df = df.replace(',', '.', regex=True)
        df.fillna(0, inplace=True)
        df.iloc[::] = df.iloc[::].astype(float)
        df.to_csv(
            output_data_path,
            index=False,
            header=True,
        )
        return df

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == '__main__':
    data_transformation()
