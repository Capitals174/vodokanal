import sys

import click
import pandas as pd

from vodokanal.exceptions import CustomException
from vodokanal.utils import load_object


@click.command()
@click.option(
    '--data',
    required=True,
    type=click.Path(exists=True),
    prompt='Specify input path',
    help='Path to input data file for prediction',
)
@click.option(
    '--preprocessor_path',
    required=True,
    type=click.Path(),
    prompt='Specify input path',
    help='Path to preprocessor',
)
@click.option(
    '--model_path',
    required=True,
    type=click.Path(),
    prompt='Specify input path',
    help='Path to trained model',
)
@click.option(
    '--predictions_path',
    required=True,
    type=click.Path(),
    prompt='Specify output path',
    help='Path to prediction values',
)
def predict(data, model_path, preprocessor_path, predictions_path):
    try:
        model = load_object(file_path=model_path)
        preprocessor = load_object(file_path=preprocessor_path)
        data_scaled = preprocessor.transform(data)
        preds = model.predict(data_scaled)
        pd.DataFrame(preds).to_csv(predictions_path)

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == '__main__':
    predict()
