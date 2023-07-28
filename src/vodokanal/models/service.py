import numpy as np
import constants
from typing import Any, Dict, Iterable, List, Tuple, Union

class Common:

    async def _cartesian(
        self,
        arrays: Iterable,
        out: Iterable = None,
    ) -> List[List[float]]:
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
            await self._cartesian(arrays[1:], out=out[0:m, 1:])
            for j in range(1, arrays[0].size):
                out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]

        return out.tolist()

    async def _save_object(self, filename: str, obj: Any):
        _file = BytesIO()
        bytes_file = pickle.dumps(obj)
        _file = BytesIO(bytes_file)

        filename = os.path.join(MODELS, filename)

        await self.minio.put(
            file=interface.File(
                data=_file,
                filename=filename,
                content_type=constants.CONTENT_TYPE,
                length=_file.getbuffer().nbytes
            )
        )



class Optimizer(Common):
    def __init__(self, reagents, ):
        self.reagents = reagents

    def _prepare_reagents(self, df):
        reagents = self.reagents_repo.get_all()

        result = {}
        for column in constants.DYNAMIC_COLUMNS:
            mdf[column]

        for reagent in reagents:
            result[reagent.name] = reagent

        return result

    @staticmethod
    def _prepare_static(data: dto.FeaturesPack) -> Dict[str, List[float]]:
        _data = [data.feculence, data.ph, data.mn, data.fe, data.alkalinity]
        custom_data_input_dict = dict(
            zip(constants.STATIC_COLUMNS, [[__] for __ in _data])
        )

        return custom_data_input_dict

    def _get_weights_and_features(
            self, data: dto.FeaturesPack, reagents: Dict[str, entities.Reagent]
    ) -> pd.DataFrame:

        custom_data_input_dict = await self._prepare_static(data)

        df_feature = pd.DataFrame(custom_data_input_dict)
        stat = await self._get_features_stat(reagents)

        w_nh4 = await self._define_weight(stat.nh4_min, stat.nh4_max)
        w_lime = await self._define_weight(stat.lime_min, stat.lime_max)
        w_sa = await self._define_weight(stat.sa_min, stat.sa_max)
        w_paa_kk = await self._define_weight(stat.paa_kk_min, stat.paa_kk_max)
        w_paa_f = await self._define_weight(stat.paa_f_min, stat.paa_f_max)
        w_pm = await self._define_weight(
            stat.permanganate_min, stat.permanganate_max
        )

        weights_data = [w_nh4, w_lime, w_paa_kk, w_paa_f, w_sa, w_pm]

        combos = await self._cartesian(weights_data)
        cost = await self._define_cost(combos, reagents)
        weights_combo = pd.DataFrame(
            data=combos, columns=constants.DYNAMIC_COLUMNS
        )
        weights_combo[constants.COST_COLUMN] = cost

        single_features_weights = df_feature.iloc[0, :].to_frame().T.merge(
            weights_combo, how=constants.MERGE_STRATEGY
        )

        return single_features_weights.sort_values(by=constants.COST_COLUMN)

    def predict(self, data: dto.FeaturesPack) -> bool:
        reagents = await self._prepare_reagents()
        results = await self._get_weights_and_features(data, reagents)
        results_without_cost = results.drop(constants.COST_COLUMN, axis=1)
        prediction = await self.predictor.predict(results_without_cost)

        results[constants.PREDICT_COLUMN] = prediction
        if (results[constants.PREDICT_COLUMN] == 1).any():
            df_true = results[results[constants.PREDICT_COLUMN] == 1].iloc[0]

            await self._add_prediction(
                data=data,
                reagents=reagents,
                predictions=df_true,
            )

            return True

    def _get_features_stat(
            reagents: Dict[str, entities.Reagent]
    ) -> dto.FeaturesStat:

        stat = {}

        for item, value in constants.MAPPER.items():
            reagent = reagents.get(value)
            if not reagent:
                continue

            min_value = reagent.min_value
            max_value = reagent.max_value

            stat[f'{item}_min'] = (
                float(min_value)
                if min_value is not None else constants.DEFAULT_MIN
            )
            stat[f'{item}_max'] = (
                float(max_value)
                if max_value is not None else constants.DEFAULT_MAX
            )

        return dto.FeaturesStat(**stat)

    def _define_weight(min_value: float, max_value: float) -> np.ndarray:
        return np.arange(
            min_value, max_value,
            (max_value - min_value) / constants.MATRIX_STEP
        )