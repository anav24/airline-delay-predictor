from pathlib import Path
import argparse

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "final_departure_delay_model.joblib"


def load_model_bundle(model_path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Run this first:\n"
            "  python src/train_final_departure_model.py"
        )

    return joblib.load(model_path)


def validate_input_columns(df, features):
    missing = [feature for feature in features if feature not in df.columns]

    if missing:
        missing_text = "\n".join(f"  - {feature}" for feature in missing)
        raise ValueError(
            "Input file is missing required model features:\n"
            f"{missing_text}"
        )


def predict(input_path, output_path, model_path):
    bundle = load_model_bundle(model_path)

    pipeline = bundle["pipeline"]
    threshold = bundle["threshold"]
    features = bundle["features"]

    df = pd.read_csv(input_path, low_memory=False)
    validate_input_columns(df, features)

    X = df[features].copy()

    probabilities = pipeline.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    output = df.copy()
    output["delay_probability"] = probabilities
    output["predicted_arrival_delay_15"] = predictions
    output["prediction_threshold"] = threshold

    output.to_csv(output_path, index=False)

    print(f"Loaded rows: {len(df):,}")
    print(f"Threshold:   {threshold:.2f}")
    print(f"Predicted delayed flights: {predictions.sum():,}")
    print(f"Saved predictions to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict whether flights will arrive at least 15 minutes late."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to a CSV file containing flight rows with the required features.",
    )

    parser.add_argument(
        "--output",
        default="predictions_departure_delay.csv",
        help="Path where the prediction CSV should be saved.",
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the trained model bundle.",
    )

    args = parser.parse_args()

    predict(
        input_path=Path(args.input),
        output_path=Path(args.output),
        model_path=Path(args.model),
    )


if __name__ == "__main__":
    main()
