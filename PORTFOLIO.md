# Used-EV Price Prediction

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## At a glance

XGBoost regressor that predicts used electric-vehicle (EV) resale prices in KRW million, built for the DACON used-EV price hackathon (December 2024 to January 2025).

**Problem.** Battery capacity, a likely value driver, is observed in only 4,786 of 7,497 training rows and 542 of 846 test rows. The test set has the same gap, so whatever fills the training gap must also work at prediction time. Depreciation also differs by manufacturer, model, condition, and drivetrain, so a simple mileage-and-age rule does not fit.

**Result.** DACON leaderboard score 0.919 (normalized RMSE, root mean squared error; lower is better).

[README](README.md) · [CLI pipeline](src/ev_price_prediction_xgb.py) · [Notebook](notebooks/EV_price_prediction_xgb.ipynb)

## Key decisions

**1. Handling missing battery capacity.**
Options: drop the rows, fill one mean or median, add a missing-value flag, or predict the value from other vehicle fields.
Choice: a separate XGBoost imputer trained only on rows with observed capacity, using make, model, condition, drivetrain, mileage, and warranty.
Why: in the notebook's ANOVA (analysis of variance) tests, capacity differed by make, model, condition, and drivetrain (p < 0.05) but not by accident history (p = 0.5151), so the other fields carry information a single average throws away; dropping rows would lose over a third of the data. The CLI fits the imputer on the development partition only, so validation values do not leak into it.
Cost: imputation errors flow into the price model.

**2. Features that describe depreciation.**
Options: raw columns only, or engineered features for usage and vehicle segment.
Choice: make x model x condition combinations, mileage per year of age, odometer km per kWh of capacity, a used flag, a drivetrain weight, and an accident flag, each created only when its source columns exist.
Why: cars of the same age differ in usage intensity and battery size, and combinations capture segment-specific depreciation.
Cost: the km-per-kWh feature uses the imputed capacity, so imputation error enters twice.

**3. Validation.**
Options: none (the competition notebook trained on all rows and relied on the leaderboard), K-fold, a split grouped by vehicle model, or one seeded holdout.
Choice for the CLI: one seeded 80/20 random holdout plus a development-partition median baseline, then a refit on all training rows for the submission.
Why: a quick local check and a reference point without the leaderboard.
Cost: a single random split has unmeasured variance and does not test unseen vehicle models.

## Results

| Model | Metric | Value | Setting |
|---|---|---:|---|
| XGBoost price model after battery imputation | Normalized RMSE | 0.919 | DACON leaderboard |

The CLI writes the 80/20 holdout RMSE and the median-price baseline to `results/metrics.json` when run with the DACON data.

## Limitations

- The notebook and the CLI differ (default vs set hyperparameters, test-row vs training-row imputer, no holdout vs holdout), and it is not known which one produced 0.919.
- One random holdout; no time, group, or external validation.
- The gain of model-based imputation over a simple fill was not measured under the same split.
- Drivetrain weights (AWD 1.2, RWD 1.1, FWD 1.0) are hand-set, not learned.

## Next steps

- Compare mean, model-based, and missing-indicator treatments under the same split.
- Validate with folds grouped by vehicle model to test generalization to models unseen in training.
- Keep the submission file, data version, and feature list with every scored run.

## Evidence

- [`src/ev_price_prediction_xgb.py`](src/ev_price_prediction_xgb.py)
- [`notebooks/EV_price_prediction_xgb.ipynb`](notebooks/EV_price_prediction_xgb.ipynb) (row counts, ANOVA tests)
