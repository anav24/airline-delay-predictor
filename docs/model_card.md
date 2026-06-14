# Model Card: Airline Arrival Delay Predictor

## Model Overview

This project contains a machine learning model that predicts whether a U.S. domestic flight will arrive at least 15 minutes late.

The final model is a departure-delay model. It uses scheduled flight information plus information available after departure or near takeoff, such as departure delay, taxi-out time, and wheels-off time.

## Prediction Target

The model predicts:

```text
ARR_DEL15
```

Where:

```text
0 = flight arrived less than 15 minutes late
1 = flight arrived 15 or more minutes late
```

## Model Type

The final model uses:

```text
HistGradientBoostingClassifier
```

The model is wrapped in a scikit-learn pipeline that includes preprocessing for numeric and categorical features.

## Data Used

The model was trained and evaluated on Bureau of Transportation Statistics U.S. domestic airline on-time performance data from:

```text
January 2025
February 2025
March 2025
```

The temporal validation setup was:

```text
Train: January 2025
Validation: February 2025
Test: March 2025
```

Cancelled and diverted flights were removed before training and evaluation.

## Final Test Performance

The final departure-delay model achieved the following March 2025 test results:

```text
Threshold: 0.40
Accuracy:  0.9452
Precision: 0.9055
Recall:    0.8043
F1-score:  0.8519
ROC-AUC:   0.9628
```

## Intended Use

This model is intended for educational and portfolio purposes.

It can be used to demonstrate:

- End-to-end machine learning workflow
- Data cleaning
- Class imbalance handling
- Threshold tuning
- Temporal validation
- Model interpretation
- Reproducible training and prediction scripts

## Limitations

This model is not a production airline operations system.

Important limitations:

- It uses actual departure-delay information, so it is not a true before-departure prediction model.
- It does not include weather data.
- It does not include airport congestion data.
- It does not include aircraft rotation or previous-leg delay data.
- It was trained on only three months of 2025 data.
- Performance may change on future months or different operating conditions.

## Ethical and Practical Considerations

The model should not be used to make passenger, staffing, or operational decisions without further validation.

Predictions should be interpreted as estimated risk scores, not guarantees.

## Reproducibility

Train the final model with:

```text
python src/train_final_departure_model.py
```

Generate predictions with:

```text
python src/predict_departure_delay.py --input path/to/new_flights.csv --output predictions_departure_delay.csv
```

The required prediction input columns are documented in:

```text
docs/prediction_input_schema.md
```
