import sys

import numpy as np
import pandas as pd

from src.vodokanal.exceptions import CustomException
from src.vodokanal.models.predict_pipeline import predict


def cartesian(arrays, out=None):
    """
    Generate a Cartesian product of input arrays.
    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m : (j + 1) * m, 1:] = out[0:m, 1:]  # noqa
    return out.tolist()


class PredictsOptimizer:
    def __init__(self):
        pass


class Optimizer:
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
        PAA_kk: int,
        PAA_f: int,
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
        self.PAA_kk = PAA_kk
        self.PAA_f = PAA_f
        self.sa = sa
        self.permanganate = permanganate

    def predict(self, pred_df):
        try:
            print("Mid Prediction")
            results = predict(pred_df)
            print("after Prediction: ", results)
            # preds = model.predict(self.get_weights_and_features())
            pred_df['pred'] = results
            if (pred_df['pred'] == 1).any():
                df_true = pred_df[pred_df['pred'] == 1]

                df_true['cost'] = (
                    df_true['sa'] * 17150
                    + df_true['permanganate'] * 295900
                    + (df_true['paa_kk'] + df_true['paa_f']) * 100000
                )
                return df_true.sort_values(by='cost').iloc[0, -6:].tolist()
            else:
                return 0

        except Exception as e:
            raise CustomException(e, sys)

    def get_weights_and_features(self):
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
                "paa_kk": [self.PAA_kk],
                "paa_f": [self.PAA_f],
                "sa": [self.sa],
                "permanganate": [self.permanganate],
            }
            df = pd.read_excel('data/raw/data_new_v1.xlsx')
            df_feature = pd.DataFrame(custom_data_input_dict)
            w_pm = np.arange(
                df['permanganate'].min(),
                df['permanganate'].max(),
                (df['permanganate'].max() - df['permanganate'].min()) / 50,
            )
            w_sa = np.arange(
                df['sa'].min(),
                df['sa'].max(),
                (df['sa'].max() - df['sa'].min()) / 50,
            )
            w_paakk = np.arange(
                df['paa_kk'].min(),
                df['paa_kk'].max(),
                (df['paa_kk'].max() - df['paa_kk'].min()) / 50,
            )
            w_paaf = np.arange(
                df['paa_f'].min(),
                df['paa_f'].max(),
                (df['paa_f'].max() - df['paa_f'].min()) / 50,
            )

            weights = pd.DataFrame(data=[w_paakk, w_paaf, w_sa, w_pm]).T
            weights.columns = ['paa_kk', 'paa_f', 'sa', 'permanganate']
            combos = cartesian([w_paakk, w_paaf, w_sa, w_pm])
            weights_combo = pd.DataFrame(
                data=combos, columns=['paa_kk', 'paa_f', 'sa', 'permanganate']
            )

            single_features_weights = (
                df_feature.iloc[0, :-4]
                .to_frame()
                .T.merge(weights_combo, how='cross')
            )

            return single_features_weights

        except Exception as e:
            raise CustomException(e, sys)
