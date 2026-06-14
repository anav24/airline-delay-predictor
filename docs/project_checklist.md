# Project Checklist

This checklist summarizes the current project status and what is intentionally kept local.

## Completed

- [x] Created a GitHub repository for the project
- [x] Added a clean project README
- [x] Added `.gitignore` rules to keep data, models, and generated files local
- [x] Completed exploratory analysis notebooks
- [x] Built baseline models
- [x] Compared multiple model types
- [x] Performed threshold tuning
- [x] Used temporal validation across January, February, and March 2025
- [x] Built a final departure-delay model
- [x] Added a reproducible final training script
- [x] Added a prediction script for new flight records
- [x] Added prediction input schema documentation
- [x] Added a model card

## Local-Only Files

These folders/files are intentionally not pushed to GitHub:

```text
data/
models/
visuals/
*.csv
*.parquet
*.joblib
*.pkl
```

This keeps large raw datasets, generated models, and local outputs out of the repository.

## Reproducible Workflow

Train the final model:

```text
python src/train_final_departure_model.py
```

Generate predictions:

```text
python src/predict_departure_delay.py --input path/to/new_flights.csv --output predictions_departure_delay.csv
```

## Current Final Model

Final departure-delay model:

```text
Model: HistGradientBoostingClassifier
Threshold: 0.40
Test Accuracy:  0.9452
Test Precision: 0.9055
Test Recall:    0.8043
Test F1-score:  0.8519
Test ROC-AUC:   0.9628
```

## Future Improvements

- [ ] Add weather data by airport and date
- [ ] Add airport congestion features
- [ ] Add aircraft rotation or previous-leg delay features
- [ ] Train on a full year of data
- [ ] Compare XGBoost or LightGBM
- [ ] Build a small Streamlit dashboard
- [ ] Add automated tests for the scripts
- [ ] Save reusable visual summaries
