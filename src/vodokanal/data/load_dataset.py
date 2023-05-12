import logging
import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.vodokanal.exceptions import CustomException


class DataLoaderConfig:
    interim_data_path: str = os.path.join(
        '../..', '..', 'data', 'interim', 'data.csv'
    )
    raw_data_path: str = os.path.join(
        '../..', '..', 'data', 'raw', 'data_new_v1.xlsx'
    )


class DataLoader:
    def __init__(self):
        self.ingestion_config = DataLoaderConfig()

    def load_data(self):
        logging.info('Entering data ingestion')
        try:
            df = pd.read_excel(self.ingestion_config.raw_data_path)
            logging.info('Reading data')

            os.makedirs(
                os.path.dirname(self.ingestion_config.interim_data_path),
                exist_ok=True,
            )

            df.to_csv(
                self.ingestion_config.interim_data_path,
                index=False,
                header=True,
            )
            logging.info('Loading data completed')

            return df

        except CustomException as e:
            raise CustomException(e, sys)


#
# if __name__ == '__main__':
#     obj = DataLoader()
#     load_data = obj.load_data()
