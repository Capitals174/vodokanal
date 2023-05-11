import os
import sys
from dataclasses import dataclass

import pandas as pd
from src.vodokanal.exceptions import CustomException


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
