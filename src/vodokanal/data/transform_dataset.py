import os
import sys
from dataclasses import dataclass

import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from src.vodokanal.exceptions import CustomException

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# from vodokanal.data.data_transformation import DataTransformation
# from vodokanal.exceptions import CustomException
# from vodokanal.logger import logging
# from vodokanal.models.model_trainer import ModelTrainer


@dataclass
class DataTransformConfig:
    processed_data_path: str = os.path.join(
        '../..', '..', 'data', 'processed', 'data.csv'
    )
    source_data_path: str = os.path.join(
        '../..', '..', 'data', 'interim', 'data.csv'
    )


class DataTransform:
    def __init__(self):
        self.transform_config = DataTransformConfig()

    def initiate_data_transformation(self):
        try:
            df = pd.read_csv(self.transform_config.source_data_path)
            df = df.replace(',', '.', regex=True)
            df.fillna(0, inplace=True)
            df.iloc[::] = df.iloc[::].astype(float)
            df.to_csv(self.transform_config.processed_data_path, index=False,
                header=True)
            return df

        except Exception as e:
            raise CustomException(e, sys)


        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    obj = DataTransform()
    data = obj.initiate_data_transformation()
