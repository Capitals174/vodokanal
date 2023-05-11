import os
import sys
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from src.vodokanal.exceptions import CustomException
from src.vodokanal.utils import save_object
import constants

class ModelCreationConfig:
    source_data_path: str = os.path.join(
        '../..', '..', 'data', 'processed', 'data.csv'
    )


class CreateModel:
    def __init__(self):
        self.model_config = ModelCreationConfig()

    def _get_data_transformer_object(self):
        try:
            numerical_columns = constants.numerical_columns

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            logging.info("Numerical columns")

            preprocessor = ColumnTransformer(
                [("num_pipeline", num_pipeline, numerical_columns)]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def preprocessing(self):
        preprocessing_obj = self._get_data_transformer_object()
        df = pd.read_csv(self.model_config.source_data_path)
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=69
        )
        target_column_name = "quality"

        input_feature_train_df = train_df.drop(
            columns=[target_column_name], axis=1
        )
        target_feature_train_df = train_df[target_column_name]

        input_feature_test_df = test_df.drop(
            columns=[target_column_name], axis=1
        )
        target_feature_test_df = test_df[target_column_name]

        logging.info(
            "Applying preprocessing object on training "
            "dataframe and testing dataframe."
        )

        input_feature_train_arr = preprocessing_obj.fit_transform(
            input_feature_train_df
        )
        input_feature_test_arr = preprocessing_obj.transform(
            input_feature_test_df
        )

        train_arr = np.c_[
            input_feature_train_arr, np.array(target_feature_train_df)
        ]
        test_arr = np.c_[
            input_feature_test_arr, np.array(target_feature_test_df)
        ]

        logging.info("Saved preprocessing object.")

        save_object(
            file_path=(
                self.data_transformation_config.preprocessor_obj_file_path
            ),
            obj=preprocessing_obj,
        )

        return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path,
        )
