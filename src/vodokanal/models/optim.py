import numpy as np
import pandas as pd
import itertools
import os
import pickle
from catboost import CatBoostClassifier
import joblib

import constants

class Optimiser:
    def __init__(self, reagent_prices, model, preprocessor, step):
        self.reagent_prices = reagent_prices
        self.model = model
        self.preprocessor = preprocessor
        self.step = step

    @staticmethod
    def create_default_restrictions(data):
        df = pd.read_csv(data)
        dynamic_columns = constants.DYNAMIC_COLUMNS
        default_restrictions = {}

        for key in dynamic_columns:
            min_value = df[key].min()
            max_value = df[key].max()
            values = (min_value, max_value)
            default_restrictions[key] = values

        return default_restrictions

    @staticmethod
    def generate_combinations(restrictions, step):
        """
        Генерация матрицы возможных комбинаций материалов
        Args:
            restrictions_on_materials - словарь из верхней и нижней границы
                по каждому материалу для построения комбинаций
        Returns:
            material_combinations_df - df из всех возможных комбинаций
                материалов
        """
        lower_bound, upper_bound = 0, 1
        all_variations_of_material = []
        for reagent_name in restrictions:
            variations_of_material = list(
                np.linspace(
                    restrictions[reagent_name][lower_bound],
                    restrictions[reagent_name][upper_bound]
                    , step
                )
            )
            all_variations_of_material.append(variations_of_material)

        material_combinations_df = pd.DataFrame(
           itertools.product(*all_variations_of_material),
            columns=restrictions.keys()
        )
        return material_combinations_df

    @staticmethod
    def create_data_for_predict(df_with_dynamic_features, origin_water_params):
        df = df_with_dynamic_features.copy()
        for key, value in origin_water_params.items():
            df[key] = value

        return df

    @staticmethod
    def get_costs(df_with_dynamic_features, water_flow):
        df = df_with_dynamic_features.copy()
        water_flow = water_flow * 1000
        prices = constants.reagent_prices
        df['cost'] = 0
        for key, value in prices.items():
            df['cost'] = df['cost'] + df[key] * value / 1000000000 * water_flow

        return df

    def get_prediction(self, data):
        df['class'] = ...
        # return df

    @staticmethod
    def _load_object(file_path):
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    def predict(self, data, model_path, preprocessor_path):
        model = self._load_object(file_path=model_path)
        preprocessor = self._load_object(file_path=preprocessor_path)
        data_scaled = preprocessor.transform(data)
        preds = model.predict(data_scaled)
        return preds

    @staticmethod
    def search_best_recommendation(data: pd.DataFrame):
        df = data[data['quality_predict'] == 1]
        df = df.sort_values(by='cost')
        best_result = df.head(1)
        return best_result


if __name__ == '__main__':
    opt = Optimiser
    data_path = os.path.join('../..', '..', 'data', 'processed', 'data.csv')
    model_path = os.path.join('../..', '..', "models", "model.pkl")
    # preprocessor_path = os.path.join('../..', '..', "models", "preprocessor.pkl")
    model = joblib.load(model_path)
    # opt.model = model
    origin_water_params = {'feculence': 42, 'ph': 7, 'mn': 0, 'fe': 2.7,
                           'alkalinity': 0.75}

    res = opt.create_default_restrictions(data=data_path)
    combimations = opt.generate_combinations(res, 10)
    df = opt.create_data_for_predict(combimations, origin_water_params)
    # TODO удалить когда будет препроцессор
    df = df[['feculence', 'ph', 'mn', 'fe', 'alkalinity', 'lime', 'paa_kk',
             'paa_f', 'sa', 'permanganate']].drop_duplicates()
    df['quality_predict'] = model.predict(df)
    df = opt.get_costs(df, 5400)
    best_recommend = opt.search_best_recommendation(df)
