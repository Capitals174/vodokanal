import os
import sys

import click
import pandas as pd

from src.vodokanal.exceptions import CustomException
from src.vodokanal.utils import load_object


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
def predict(data, model_path, preprocessor_path):
    try:
        model = load_object(file_path=model_path)
        preprocessor = load_object(file_path=preprocessor_path)
        print("After Loading")
        data_scaled = preprocessor.transform(data)
        preds = model.predict(data_scaled)
        return preds

    except Exception as e:
        raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        chromasity: int,
        feculence: int,
        ph: int,
        mn: int,
        fe: int,
        alkalinity: int,
        nh4: int,
        lime: int,
        paa_kk: int,
        paa_f: int,
        sa: int,
        permanganate: int,
    ):
        self.chromasity = chromasity
        self.feculence = feculence
        self.ph = ph
        self.mn = mn
        self.fe = fe
        self.alkalinity = alkalinity
        self.nh4 = nh4
        self.lime = lime
        self.paa_kk = paa_kk
        self.paa_f = paa_f
        self.sa = sa
        self.permanganate = permanganate

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "chromasity": [self.chromasity],
                "feculence": [self.feculence],
                "ph": [self.ph],
                "mn": [self.mn],
                "fe": [self.fe],
                "alkalinity": [self.alkalinity],
                "nh4": [self.nh4],
                "lime": [self.lime],
                "PAA_kk": [self.PAA_kk],
                "PAA_f": [self.PAA_f],
                "sa": [self.sa],
                "permanganate": [self.permanganate],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    predict()
