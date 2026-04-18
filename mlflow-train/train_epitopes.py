# mlflow-train/train_epitopes.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

from feature_helpers import (
    sequence_to_bcell_features,
    sequence_to_tcell_features
)


def train_bcell_model(
    df: pd.DataFrame,
    output_model_name: str = "BcellPred",
    output_version: str = "1"
) -> Dict[str, Any]:
    # df: колонки ["sequence", "is_bcell_epitope"]
    X = np.array([sequence_to_bcell_features(s) for s in df["sequence"].values])
    y = df["is_bcell_epitope"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    with mlflow.start_run():
        mlflow.set_tag("task", "bcell_epitope_prediction")
        mlflow.set_tag("origin", "IEDB_like")

        mlflow.log_params({"model_type": "RandomForest", "n_estimators": 100})
        mlflow.log_metric("test_auc", auc)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=output_model_name,
            signature=mlflow.models.infer_signature(X_train, model.predict(X_train))
        )

    return {"model_name": output_model_name, "test_auc": auc}


def train_tcell_model(
    df: pd.DataFrame,
    output_model_name: str = "TcellPred",
    output_version: str = "1"
) -> Dict[str, Any]:
    # df: колонки ["sequence", "is_tcell_epitope"]
    X = np.array([sequence_to_tcell_features(s) for s in df["sequence"].values])
    y = df["is_tcell_epitope"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    with mlflow.start_run():
        mlflow.set_tag("task", "tcell_epitope_prediction")
        mlflow.set_tag("origin", "IEDB_like")

        mlflow.log_params({"model_type": "RandomForest", "n_estimators": 100})
        mlflow.log_metric("test_auc", auc)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=output_model_name,
            signature=mlflow.models.infer_signature(X_train, model.predict(X_train))
        )

    return {"model_name": output_model_name, "test_auc": auc}


if __name__ == "__main__":
    # df_bcell = ...  # B‑cell эпитопы
    # df_tcell = ...  # T‑cell эпитопы
    # train_bcell_model(df_bcell, "BcellPred", "1")
    # train_tcell_model(df_tcell, "TcellPred", "1")
    pass
