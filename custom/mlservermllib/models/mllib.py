from mlserver import MLModel, types
from mlserver.errors import InferenceError
from .utils import get_mllib_load
from pyspark import SparkContext, SparkConf


class MLLibModel(MLModel):

    async def load(self) -> bool:
        conf = SparkConf().set('spark.driver.host', '127.0.0.1')
        sc = SparkContext(appName="MLLibModel", conf=conf)

        model_uri = self._settings.parameters.uri
        model_load = await get_mllib_load(self._settings)

        self._model = model_load(sc, model_uri)

        self.ready = True
        return self.ready

    async def predict(self, payload: types.InferenceRequest) -> types.InferenceResponse:
        payload = self._check_request(payload)
        prediction = self._model.predict(payload.inputs[0].data)

        return types.InferenceResponse(
            model_name=self.name,
            model_version=self.version,
            outputs=[
                types.ResponseOutput(
                    name="predict",
                    shape=[1],
                    datatype="FP32",
                    data=prediction,
                )
            ],
        )

    def _check_request(self, payload: types.InferenceRequest) -> types.InferenceRequest:
        if len(payload.inputs) != 1:
            raise InferenceError(
                "MLLibModel only supports a single input tensor "
                f"({len(payload.inputs)} were received)"
            )

        return payload