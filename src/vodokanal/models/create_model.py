import logging
import sys

import click
import constants
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.vodokanal.exceptions import CustomException
from src.vodokanal.utils import evaluate_models, save_object


def _get_data_transformer_object():
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


def _preprocessing(input_data_path, preprocessor_path):
    preprocessing_obj = _get_data_transformer_object()
    df = pd.read_csv(input_data_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=69)
    target_column_name = "quality"

    input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
    target_feature_train_df = train_df[target_column_name]

    input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
    target_feature_test_df = test_df[target_column_name]

    logging.info(
        "Applying preprocessing object on training "
        "dataframe and testing dataframe."
    )

    input_feature_train_arr = preprocessing_obj.fit_transform(
        input_feature_train_df
    )
    input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

    train_arr = np.c_[
        input_feature_train_arr, np.array(target_feature_train_df)
    ]
    test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

    logging.info("Saved preprocessing object.")
    save_object(
        file_path=(preprocessor_path),
        obj=preprocessing_obj,
    )

    return (train_arr, test_arr)


@click.command()
@click.option(
    '--input_data_path',
    required=True,
    type=click.Path(exists=True),
    prompt='Specify input path',
    help='Path to input data file',
)
@click.option(
    '--preprocessor_path',
    required=True,
    type=click.Path(),
    prompt='Specify output path',
    help='Path to save preprocessor',
)
@click.option(
    '--model_path',
    required=True,
    type=click.Path(),
    prompt='Specify output path',
    help='Path to save trained model',
)
def train_model(input_data_path, preprocessor_path, model_path):
    try:
        train_array, test_array = _preprocessing(
            input_data_path, preprocessor_path
        )
        logging.info("Split training and test input data")
        X_train, y_train, X_test, y_test = (
            train_array[:, :-1],
            train_array[:, -1],
            test_array[:, :-1],
            test_array[:, -1],
        )
        models = {
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Linear Classifier": SGDClassifier(),
            "CatBoosting Classifier": CatBoostClassifier(verbose=False),
            "AdaBoost Classifier": AdaBoostClassifier(),
        }
        params = {
            "Random Forest": {'n_estimators': [128, 256]},
            "Gradient Boosting": {
                'learning_rate': [0.1, 0.001],
                'subsample': [0.8, 0.85, 0.9],
                'n_estimators': [128, 256],
            },
            "Linear Classifier": {},
            "CatBoosting Classifier": {
                'depth': [6, 10],
                'learning_rate': [0.01, 0.1],
                'iterations': [30, 100],
            },
            "AdaBoost Classifier": {
                'learning_rate': [0.1, 0.01],
                'n_estimators': [64, 128, 256],
            },
        }

        model_report: dict = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=models,
            param=params,
        )

        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]
        print(best_model)

        if best_model_score < 0.6:
            raise CustomException("No best model found")
        logging.info('Best found model on both training and testing dataset')

        save_object(
            file_path=model_path,
            obj=best_model,
        )

        predicted = best_model.predict(X_test)

        precision_score_ = precision_score(y_test, predicted)
        return precision_score_

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == '__main__':
    train_model()
