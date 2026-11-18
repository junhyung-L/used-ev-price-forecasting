# Used-EV Price Forecasting: Separating Battery Imputation from Price Regression

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## Summary

For the DACON used-EV price hackathon (December 2024 to January 2025), this
project predicts resale prices in KRW million. Battery capacity, a likely
value driver, is missing in about 36% of rows, so it is treated as a prediction
problem of its own before the XGBoost price regressor is trained. The
user-confirmed DACON normalized leaderboard RMSE is **0.919**. The notebook
workflow was later packaged as a command-line pipeline with a seeded holdout
and a median baseline.

## Challenge

Used-EV value is not a simple mileage-and-age rule. Depreciation differs by
manufacturer, model, condition, and drivetrain, and the saved notebook output
shows battery capacity observed in only 4,786 of 7,497 training rows and 542 of
846 test rows. The test set has the same gap, so whatever fills the training
gap must also work at prediction time.

## Decisions

**1. How to handle missing battery capacity.**
Options: drop the rows, fill one mean or median, add a missing-value flag, or
predict the value from other vehicle fields. Choice: a separate XGBoost
imputer trained only on rows with observed capacity, using make, model,
condition, drivetrain, mileage, and warranty. Why: in the notebook's ANOVA
tests, capacity differed by make, model, condition, and drivetrain (p<0.05)
but not by accident history (p = 0.52), so the other fields carry information
a single average throws away, and dropping rows would lose over a third of the
data. In the CLI the imputer is fit on the development partition only, so
validation values do not leak into it. Cost: imputation errors flow into the
price model, and the gain over a simple fill was never measured under the same
split.

**2. Which features describe depreciation.**
Options: raw columns only, or engineered features for usage and vehicle
segment. Choice: make x model x condition combinations, mileage per year of
age, odometer km per kWh of capacity, a used flag, a drivetrain weight, and an
accident flag, each created only when its source columns exist. Why: cars of
the same age can differ in usage intensity and battery size, and combinations
capture segment-specific depreciation. Cost: the km-per-kWh feature uses the
imputed capacity, so imputation error enters twice; the drivetrain weights
(AWD 1.2, RWD 1.1, FWD 1.0) are hand-set rather than learned.

**3. How to validate.**
Options: none (the competition notebook trained on all rows and relied on the
leaderboard), K-fold, a split grouped by vehicle model, or one seeded holdout.
Choice for the CLI: one seeded 80/20 random holdout plus a
development-partition median baseline, then a refit on all training rows for
the submission. Why: it gives a quick local check and a reference point
without the leaderboard. Cost: a single split has unmeasured variance, and a
random split does not test unseen vehicle models.

## Results

| Model | Metric | Value | Setting |
|---|---|---:|---|
| XGBoost price model after battery imputation | DACON normalized RMSE | 0.919 | Competition leaderboard, user-confirmed |
| Development-partition median price (baseline) | RMSE | - | CLI 80/20 holdout |
| XGBoost price model (CLI) | RMSE | - | CLI 80/20 holdout |

"-": produced in `results/metrics.json` only when the DACON data is supplied;
no run output is retained.

## What the result does not show

- The original data, submission, and leaderboard capture are not retained, so
  0.919 cannot be recomputed here, and a new local holdout RMSE is not
  directly comparable with it.
- The notebook and the CLI differ (default versus set hyperparameters, a
  test-row imputer versus a training-row imputer, no holdout versus a
  holdout). Which configuration produced 0.919 is not recorded.
- One random holdout is not time, group, or external validation.

## What I would do next

Retain the submission file, data version, and feature list with every scored
run; compare mean, model-based, and missing-indicator treatments under the
same split; and validate with folds grouped by vehicle model to test
generalisation to models unseen in training.

## Evidence

- [`src/ev_price_prediction_xgb.py`](src/ev_price_prediction_xgb.py)
- [`notebooks/EV_price_prediction_xgb.ipynb`](notebooks/EV_price_prediction_xgb.ipynb)
- [`docs/PROJECT_REVIEW.md`](docs/PROJECT_REVIEW.md)
