import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from vodokanal.data.data_transformation import DataTransformation
from vodokanal.exceptions import CustomException
from vodokanal.logger import logging
from vodokanal.models.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('data', 'processed', 'data.csv')
    train_data_path: str = os.path.join('data', 'processed', 'train.csv')
    test_data_path: str = os.path.join('data', 'processed', 'test.csv')
    source_data_path: str = os.path.join(
        '../..', '..', 'data', 'raw', 'data_new_v1.xlsx'
    )


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entering data ingestion')
        try:
            df = pd.read_excel(self.ingestion_config.source_data_path)
            logging.info('Reading data')

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )

            df.to_csv(
                self.ingestion_config.raw_data_path, index=False, header=True
            )
            logging.info('Train test split initalization')

            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=69
            )

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logging.info('Data Ingesttion completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except CustomException as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
