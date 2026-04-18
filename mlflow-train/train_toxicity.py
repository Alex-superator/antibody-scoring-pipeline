# mlflow-train/train_toxicity.py
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from typing import Dict, Any

from feature_helpers import (
    make_plm_embeddings,
    sequence_to_features_toxicity
)


def train_toxicity_model(
    df: pd.DataFrame,
    output_model_name: str = "StrucToxNet",
    output_version: str = "1"
) -> Dict[str, Any]:
    # 1. Features
    sequences = df["sequence"].values
    # В реальности: либо PLM (ProtT5‑like), либо handcrafted features
    X = np.array([sequence_to_features_toxicity(s) for s in sequences])
    y = df["is_toxic"].values

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Model
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # 4. Metrics
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    # 5. Log to MLflow
    with mlflow.start_run():
        mlflow.set_tag("task", "toxicity_classification")
        mlflow.set_tag("sota_origin", "StrucToxNet_like")

        mlflow.log_params({
            "model_type": "RandomForest",
            "n_estimators": 100,
            "test_size": 0.2,
        })

        mlflow.log_metric("test_auc", auc)

        # Log model to MLflow as pyfunc and to Registry
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=output_model_name,
            signature=mlflow.models.infer_signature(
                X_train, model.predict(X_train)
            ),
        )

    return {"model_name": output_model_name, "test_auc": auc, "run_id": mlflow.last_active_run_id()}


if __name__ == "__main__":
    # типичный вызов (для демо можно сгенерировать данные)
    # df = pd.read_csv("data/toxicity_dataset.csv")
    # train_toxicity_model(df, output_model_name="StrucToxNet", output_version="1")
    pass
