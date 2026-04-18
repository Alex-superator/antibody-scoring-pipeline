# mlflow-train/train_allergenicity.py
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

from feature_helpers import (
    make_plm_embeddings,
    sequence_to_features_allergenicity
)


def train_allergenicity_model(
    df: pd.DataFrame,
    output_model_name: str = "AllergenAI",
    output_version: str = "1"
) -> Dict[str, Any]:
    sequences = df["sequence"].values
    X = np.array([sequence_to_features_allergenicity(s) for s in sequences])
    y = df["is_allergen"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    with mlflow.start_run():
        mlflow.set_tag("task", "allergenicity_classification")
        mlflow.set_tag("origin", "AllergenAI_like")

        mlflow.log_params({
            "model_type": "RandomForest",
            "n_estimators": 100,
        })

        mlflow.log_metric("test_auc", auc)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=output_model_name,
            signature=mlflow.models.infer_signature(X_train, model.predict(X_train))
        )

    return {"model_name": output_model_name, "test_auc": auc}


if __name__ == "__main__":
    # df = pd.read_csv("data/allergenicity_dataset.csv")
    # train_allergenicity_model(df, "AllergenAI", "1")
    pass
