from typing import Dict
from mlflow.pyfunc import PyFuncModel
from mlflow import pyfunc
import logging

logger = logging.getLogger(__name__)


def load_mlflow_model(model_name: str, model_version: str) -> PyFuncModel:
    uri = f"models:/{model_name}/{model_version}"
    logger.info(f"Loading MLflow model: {uri}")
    model = pyfunc.load_model(uri)
    return model


class MLflowModelRegistry:
    def __init__(self):
        self.toxicity_model: PyFuncModel | None = None
        self.allergenicity_model: PyFuncModel | None = None
        self.bcell_model: PyFuncModel | None = None
        self.tcell_model: PyFuncModel | None = None
        self.stability_model: PyFuncModel | None = None

    def load_models(self, env: Dict[str, str]):
        self.toxicity_model = load_mlflow_model(
            env["TOXICITY_MODEL_NAME"], env["TOXICITY_MODEL_VERSION"]
        )
        self.allergenicity_model = load_mlflow_model(
            env["ALLERGENICITY_MODEL_NAME"], env["ALLERGENICITY_MODEL_VERSION"]
        )
        self.bcell_model = load_mlflow_model(
            env["BCELL_MODEL_NAME"], env["BCELL_MODEL_VERSION"]
        )
        self.tcell_model = load_mlflow_model(
            env["TCELL_MODEL_NAME"], env["TCELL_MODEL_VERSION"]
        )
        self.stability_model = load_mlflow_model(
            env["STABILITY_MODEL_NAME"], env["STABILITY_MODEL_VERSION"]
        )


registry = MLflowModelRegistry()
