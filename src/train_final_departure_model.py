from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = [
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "data" / "raw",
]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


TARGET = "ARR_DEL15"

NUMERIC_FEATURES = [
    "MONTH",
    "DAY_OF_MONTH",
    "DAY_OF_WEEK",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "DISTANCE",
    "DISTANCE_GROUP",
    "DEP_DELAY",
    "DEP_DELAY_NEW",
    "DEP_DEL15",
    "DEP_DELAY_GROUP",
    "TAXI_OUT",
    "WHEELS_OFF",
]

CATEGORICAL_FEATURES = [
    "OP_UNIQUE_CARRIER",
    "ORIGIN",
    "DEST",
    "DEP_TIME_BLK",
    "ARR_TIME_BLK",
]


def find_departure_data_files():
    files = []

    for data_dir in DATA_DIRS:
        if not data_dir.exists():
            continue

        files.extend(sorted(data_dir.glob("*departure*.csv")))
        files.extend(sorted(data_dir.glob("*departure*.parquet")))

    if not files:
        raise FileNotFoundError(
            "No departure-delay CSV or Parquet files found. "
            "Expected files like data/raw/flights_2025_01_departure.csv."
        )

    print("Using departure-delay files only:")
    for path in files:
        print(f"  - {path.name}")

    return files


def load_file(path):
    print(f"Loading: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file type: {path}")


def load_dataset():
    files = find_departure_data_files()
    frames = [load_file(path) for path in files]

    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(col).strip() for col in df.columns]

    print(f"Loaded rows: {len(df):,}")
    print(f"Loaded columns: {len(df.columns):,}")

    return df


def clean_dataset(df):
    if "CANCELLED" in df.columns:
        df = df[pd.to_numeric(df["CANCELLED"], errors="coerce").fillna(0) == 0]

    if "DIVERTED" in df.columns:
        df = df[pd.to_numeric(df["DIVERTED"], errors="coerce").fillna(0) == 0]

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].notna()].copy()
    df[TARGET] = df[TARGET].astype(int)

    available_numeric = [col for col in NUMERIC_FEATURES if col in df.columns]
    available_categorical = [col for col in CATEGORICAL_FEATURES if col in df.columns]

    if not available_numeric and not available_categorical:
        raise ValueError("None of the expected model features were found.")

    for col in available_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in available_categorical:
        df[col] = df[col].astype("string")

    features = available_numeric + available_categorical

    df = df[features + [TARGET]].copy()

    print(f"Clean rows: {len(df):,}")
    print(f"Late rate: {df[TARGET].mean():.2%}")
    print(f"Features used: {features}")

    return df, available_numeric, available_categorical, features


def split_dataset(df):
    if "MONTH" in df.columns and {1, 2, 3}.issubset(set(df["MONTH"].dropna().astype(int).unique())):
        train_df = df[df["MONTH"] == 1].copy()
        val_df = df[df["MONTH"] == 2].copy()
        test_df = df[df["MONTH"] == 3].copy()

        split_description = {
            "method": "temporal",
            "train": "January 2025",
            "validation": "February 2025",
            "test": "March 2025",
        }
    else:
        train_val_df, test_df = train_test_split(
            df,
            test_size=0.20,
            random_state=42,
            stratify=df[TARGET],
        )

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=0.25,
            random_state=42,
            stratify=train_val_df[TARGET],
        )

        split_description = {
            "method": "random_stratified",
            "train": "60%",
            "validation": "20%",
            "test": "20%",
        }

    print("Split summary:")
    print(f"  Train rows:      {len(train_df):,}")
    print(f"  Validation rows: {len(val_df):,}")
    print(f"  Test rows:       {len(test_df):,}")

    return train_df, val_df, test_df, split_description


def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def evaluate_predictions(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if len(set(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = None

    return metrics


def choose_threshold(y_true, y_proba):
    thresholds = np.arange(0.05, 0.96, 0.01)

    results = []
    for threshold in thresholds:
        metrics = evaluate_predictions(y_true, y_proba, threshold)
        results.append(metrics)

    best = max(results, key=lambda row: row["f1"])
    return best["threshold"], results


def print_metrics(title, metrics):
    print()
    print(title)
    print("-" * len(title))
    print(f"Threshold: {metrics['threshold']:.2f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")

    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")


def main():
    df = load_dataset()
    df, numeric_features, categorical_features, features = clean_dataset(df)

    train_df, val_df, test_df, split_description = split_dataset(df)

    X_train = train_df[features]
    y_train = train_df[TARGET]

    X_val = val_df[features]
    y_val = val_df[TARGET]

    X_test = test_df[features]
    y_test = test_df[TARGET]

    pipeline = build_pipeline(numeric_features, categorical_features)

    print()
    print("Training final departure-delay model...")
    pipeline.fit(X_train, y_train)

    val_proba = pipeline.predict_proba(X_val)[:, 1]
    best_threshold, threshold_results = choose_threshold(y_val, val_proba)

    test_proba = pipeline.predict_proba(X_test)[:, 1]

    val_metrics = evaluate_predictions(y_val, val_proba, best_threshold)
    test_metrics = evaluate_predictions(y_test, test_proba, best_threshold)

    print_metrics("Validation Metrics", val_metrics)
    print_metrics("Test Metrics", test_metrics)

    bundle = {
        "pipeline": pipeline,
        "threshold": best_threshold,
        "target": TARGET,
        "features": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "split_description": split_description,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    model_path = MODELS_DIR / "final_departure_delay_model.joblib"
    metrics_path = MODELS_DIR / "final_departure_delay_metrics.json"

    joblib.dump(bundle, model_path)

    metrics_payload = {
        "model_path": str(model_path),
        "threshold": best_threshold,
        "split_description": split_description,
        "features": features,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "threshold_search_results": threshold_results,
    }

    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Saved model bundle: {model_path}")
    print(f"Saved metrics:      {metrics_path}")


if __name__ == "__main__":
    main()
