# U.S. Airline Delay Prediction with Machine Learning

## Project Overview

This project uses U.S. domestic airline on-time performance data to predict whether a flight will arrive at least 15 minutes late. The project compares two different prediction settings:

1. **Scheduled-only prediction** - predicting arrival delay before the flight departs using scheduled flight information.
2. **Departure-delay prediction** - predicting arrival delay after departure or near takeoff using actual departure-delay information.

The goal of the project is not only to build a high-performing model, but also to understand how prediction quality changes depending on what information is available at prediction time.

## Dataset

The data comes from the Bureau of Transportation Statistics airline on-time performance dataset.

The project uses January, February, and March 2025 domestic flight records.

The target variable is:

```text
ARR_DEL15
```

Where:

```text
0 = flight arrived less than 15 minutes late
1 = flight arrived 15 or more minutes late
```

Cancelled and diverted flights were removed because they do not represent normal completed-flight arrival-delay outcomes.

## Project Structure

```text
airline-delay-predictor/
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_model_comparison.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_multi_month_model.ipynb
│   ├── 06_temporal_validation.ipynb
│   ├── 07_departure_delay_model.ipynb
│   └── 08_model_interpretation.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── models/
└── visuals/
```

Note: `data/`, `models/`, and `visuals/` are kept local and are not pushed to GitHub.

## Exploratory Analysis

After removing cancelled and diverted flights, the first month of data contained 522,269 completed flights.

January 2025 target balance:

```text
Not late: 81.21%
Late:     18.79%
```

This shows that the dataset is imbalanced. A model could predict every flight as not late and still achieve about 81% accuracy, so accuracy alone is not a reliable evaluation metric.

The exploratory analysis also showed that delay rates vary by airline, origin airport, departure time block, day of week, and route.

## Modeling Approach

The project used several models:

- DummyClassifier
- Logistic Regression
- Balanced Logistic Regression
- Random Forest
- HistGradientBoostingClassifier

Evaluation metrics included:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

F1-score was emphasized because the dataset is imbalanced and the goal is to identify delayed flights, not simply predict the majority class.

## Scheduled-Only Model

The scheduled-only model used information available before departure, such as:

```text
MONTH
DAY_OF_MONTH
DAY_OF_WEEK
CRS_DEP_TIME
CRS_ARR_TIME
DISTANCE
DISTANCE_GROUP
OP_UNIQUE_CARRIER
ORIGIN
DEST
DEP_TIME_BLK
ARR_TIME_BLK
```

It did not use arrival delay, cancellation, diversion, or delay-cause columns as input features because those would leak information from after the flight outcome was known.

The best January random-split model was a HistGradientBoosting model with threshold tuning.

Best January random-split result:

```text
Model: HistGradientBoosting
Threshold: 0.23
Accuracy:  0.7547
Precision: 0.3919
Recall:    0.5542
F1:        0.4591
```

However, a more realistic temporal validation test showed that scheduled-only performance dropped when predicting a future month.

Temporal validation setup:

```text
Train: January 2025
Validation: February 2025
Test: March 2025
```

Scheduled-only March test result:

```text
Threshold: 0.10
Accuracy:  0.3345
Precision: 0.2040
Recall:    0.8258
F1:        0.3271
ROC-AUC:   0.5302
```

This showed that scheduled flight information alone did not generalize well to future-month delay prediction.

## Departure-Delay Model

The departure-delay model used information available after departure or near takeoff, including:

```text
DEP_DELAY
DEP_DELAY_NEW
DEP_DEL15
DEP_DELAY_GROUP
TAXI_OUT
WHEELS_OFF
```

This model answers a different question:

> After a flight has departed, can we predict whether it will arrive at least 15 minutes late?

The same temporal validation setup was used:

```text
Train: January 2025
Validation: February 2025
Test: March 2025
```

Departure-delay March test result:

```text
Model: HistGradientBoosting
Threshold chosen on February: 0.40
Accuracy:  0.9452
Precision: 0.9055
Recall:    0.8043
F1:        0.8519
ROC-AUC:   0.9628
```

## Model Comparison

| Model | Prediction Timing | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Scheduled-only HGB | Before departure | 0.3345 | 0.2040 | 0.8258 | 0.3271 | 0.5302 |
| Departure-delay HGB | After departure / near takeoff | 0.9452 | 0.9055 | 0.8043 | 0.8519 | 0.9628 |

The departure-delay model performed much better because actual departure delay is highly predictive of arrival delay.

## Model Interpretation

Permutation importance showed that `DEP_DELAY` was by far the most important feature.

Top features from permutation importance:

```text
DEP_DELAY
TAXI_OUT
DISTANCE
OP_UNIQUE_CARRIER
WHEELS_OFF
ORIGIN
ARR_TIME_BLK
```

The model interpretation also showed that arrival-delay rates increased sharply as departure-delay group increased. Flights that departed early or close to schedule had low arrival-delay rates, while flights with larger departure delays were very likely to arrive at least 15 minutes late.

This confirmed that the departure-delay model performs well because it uses strong near-real-time information.

## Key Lessons

This project showed several important machine learning lessons:

1. Accuracy can be misleading when the target class is imbalanced.
2. Dummy baselines are important for understanding whether a model is actually useful.
3. Class weighting changes model behavior by increasing recall for the minority class.
4. Threshold tuning can improve the precision-recall balance.
5. Random train/test splits can overestimate performance when real-world prediction is time-based.
6. Temporal validation gives a more honest estimate of future performance.
7. Model performance depends heavily on what information is available at prediction time.
8. Additional features do not always improve performance.
9. Departure delay is a much stronger signal than scheduled flight information alone.

## Limitations

The scheduled-only model had weak future-month performance. It did not include important external signals such as:

- Weather conditions
- Airport congestion
- Previous-leg aircraft delay
- Air traffic control conditions
- Real-time operational disruptions

The departure-delay model performed much better, but it is not a true pre-departure prediction model because it uses information only available after the flight has already started departing.

## Future Improvements

Possible future improvements include:

- Add weather data by airport and date
- Add airport congestion features
- Add previous-flight or aircraft rotation delay features
- Train on a full year of data
- Compare XGBoost or LightGBM models
- Build a small Streamlit dashboard for model results
- Save the final model as a reusable pipeline
- Create a prediction script for new flight records

## Reproducible Scripts

The final departure-delay model can be retrained from the local data files with:

```text
python src/train_final_departure_model.py
```

This script uses only the departure-delay modeling files:

```text
flights_2025_01_departure.csv
flights_2025_02_departure.csv
flights_2025_03_departure.csv
```

It saves the trained model and metrics locally under `models/`.

After training, predictions can be generated for a new CSV file with:

```text
python src/predict_departure_delay.py --input path/to/new_flights.csv --output predictions_departure_delay.csv
```

The prediction output includes:

```text
delay_probability
predicted_arrival_delay_15
prediction_threshold
```

## Tools Used

```text
Python
pandas
NumPy
scikit-learn
matplotlib
JupyterLab
```

## Summary

The final model comparison showed that scheduled-only flight information had limited predictive power for future-month arrival delays. In contrast, the departure-delay model achieved strong performance by using actual departure-delay information.

The final departure-delay model achieved:

```text
Accuracy:  94.52%
Precision: 90.55%
Recall:    80.43%
F1-score:  85.19%
ROC-AUC:   96.28%
```

This project demonstrates a complete machine learning workflow: data exploration, cleaning, baseline modeling, model comparison, threshold tuning, temporal validation, feature interpretation, and final evaluation.
