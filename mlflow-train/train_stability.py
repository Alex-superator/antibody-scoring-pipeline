# mlflow-train/train_stability.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

from feature_helpers import (
    sequence_to_stability_features
)


def train_stability_model(
    df: pd.DataFrame,
    output_model_name: str = "StabilityOracle",
    output_version: str = "1"
) -> Dict[str, Any]:
    # df: ["sequence", "instability_score"] (или delta‑pH, half‑life, etc.)
    X = np.array([sequence_to_stability_features(s) for s in df["sequence"].values])
    y = df["instability_score"].values  # или 1/instability, или stability_class

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # В реальности: Regression или Binary classification (unstable / stable)
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    with mlflow.start_run():
        mlflow.set_tag("task", "protein_stability_prediction")
        mlflow.set_tag("origin", "Stability_Oracle_like")

        mlflow.log_params({"model_type": "RandomForestRegressor", "n_estimators": 100})
        mlflow.log_metric("test_mse", mse)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=output_model_name,
            signature=mlflow.models.infer_signature(X_train, y_pred[:10])
        )

    return {"model_name": output_model_name, "test_mse": mse}


if __name__ == "__main__":
    # df = pd.read_csv("data/stability_dataset.csv")
    # train_stability_model(df, "StabilityOracle", "1")
    pass
