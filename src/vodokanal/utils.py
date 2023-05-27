import os
import pickle
import sys

import mlflow as mlflow
import numpy as np
from mlflow.models import infer_signature
from sklearn.metrics import precision_score, mean_squared_error, \
    mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

from vodokanal.exceptions import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for ((model_name, model), model_params) \
                in zip(models.items(), param.values()):

            # Should be provided:
            # AWS_ACCESS_KEY_ID
            # AWS_S3_BUCKET
            # AWS_SECRET_ACCESS_KEY
            # MLFLOW_S3_ENDPOINT_URL
            # MLFLOW_TRACKING_URI
            with mlflow.start_run(run_name=model_name):
                gs = GridSearchCV(model, model_params, cv=2)
                gs.fit(X_train, y_train)

                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

                y_test_pred = model.predict(X_test)
                test_model_score = precision_score(y_test, y_test_pred)

                signature = infer_signature(X_test, y_test_pred)
                mlflow.sklearn.log_model(model, "model", signature=signature)
                mlflow.log_params(gs.best_params_)
                mlflow.log_metrics({
                    "rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
                    "mae": mean_absolute_error(y_test, y_test_pred),
                    "r2": r2_score(y_test, y_test_pred)
                })

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
