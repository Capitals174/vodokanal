import logging
import os
import sys

import click
import pandas as pd

from vodokanal.exceptions import CustomException


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
def excel_to_csv(input_data_path, output_data_path):
    logging.info('Entering data ingestion')
    try:
        df = pd.read_excel(input_data_path)
        logging.info('Reading data')

        os.makedirs(
            os.path.dirname(output_data_path),
            exist_ok=True,
        )
        df.to_csv(
            output_data_path,
            index=False,
            header=True,
        )
        logging.info('Loading data completed')

        return df

    except CustomException as e:
        raise CustomException(e, sys)


if __name__ == '__main__':
    excel_to_csv()
