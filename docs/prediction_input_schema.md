# Prediction Input Schema

This document describes the required input columns for `src/predict_departure_delay.py`.

The prediction script uses the final departure-delay model. It predicts whether a completed or departing flight is likely to arrive at least 15 minutes late.

## Required Columns

The input CSV must contain these columns:

| Column | Type | Description |
|---|---|---|
| `MONTH` | numeric | Month number of the flight. |
| `DAY_OF_MONTH` | numeric | Day of the month. |
| `DAY_OF_WEEK` | numeric | Day of week encoded in the BTS dataset. |
| `CRS_DEP_TIME` | numeric | Scheduled departure time. |
| `CRS_ARR_TIME` | numeric | Scheduled arrival time. |
| `DISTANCE` | numeric | Flight distance in miles. |
| `DISTANCE_GROUP` | numeric | BTS distance group. |
| `DEP_DELAY` | numeric | Actual departure delay in minutes. |
| `DEP_DELAY_NEW` | numeric | Non-negative departure delay in minutes. |
| `DEP_DEL15` | numeric | Whether departure was delayed at least 15 minutes. |
| `DEP_DELAY_GROUP` | numeric | BTS departure delay group. |
| `TAXI_OUT` | numeric | Taxi-out time in minutes. |
| `WHEELS_OFF` | numeric | Wheels-off time. |
| `OP_UNIQUE_CARRIER` | categorical | Airline carrier code. |
| `ORIGIN` | categorical | Origin airport code. |
| `DEST` | categorical | Destination airport code. |
| `DEP_TIME_BLK` | categorical | Scheduled departure time block. |
| `ARR_TIME_BLK` | categorical | Scheduled arrival time block. |

## Target Column

The prediction input does **not** need to include `ARR_DEL15`.

`ARR_DEL15` is the training target and represents whether the flight arrived at least 15 minutes late.

## Output Columns

The prediction script adds these columns to the output CSV:

| Column | Description |
|---|---|
| `delay_probability` | Model-estimated probability that the flight arrives at least 15 minutes late. |
| `predicted_arrival_delay_15` | Final binary prediction. `1` means predicted late, `0` means predicted not late. |
| `prediction_threshold` | Probability threshold used to convert probability into the binary prediction. |

## Example Command

```text
python src/predict_departure_delay.py --input data/processed/sample_prediction_input.csv --output data/processed/sample_predictions.csv
```

## Notes

This is a departure-delay model, not a true pre-departure model. It uses information such as actual departure delay, taxi-out time, and wheels-off time.

For a true before-departure prediction, the departure-delay columns should not be used because they are not known before the flight departs.
