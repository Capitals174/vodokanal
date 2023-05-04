import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

from src.exceptions import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], train_array[:, -1], test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Linear Classifier": SGDClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }
            params = {
                "Random Forest": {
                    'n_estimators': [128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.001],
                    'subsample': [0.8, 0.85, 0.9],
                    'n_estimators': [128, 256]
                },
                "Linear Classifier": {},
                "CatBoosting Classifier": {
                    'depth': [6, 10],
                    'learning_rate': [0.01, 0.1],
                    'iterations': [30, 100]
                },
                "AdaBoost Classifier": {
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [64, 128, 256]
                }
            }

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            best_model_score = max(sorted(model_report.values()))

            best_model = models["CatBoosting Classifier"]
            print(best_model)
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(
                'Best found model on both training and testing dataset'
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            accuracy_score_ = accuracy_score(y_test, predicted)
            return accuracy_score_

        except Exception as e:
            raise CustomException(e, sys)
