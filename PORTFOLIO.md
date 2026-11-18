# Used-EV Price Forecasting: Separating Battery Imputation from Price Regression

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## Summary

Predicts used electric-vehicle (EV) resale prices in KRW million for the DACON
used-EV price hackathon (December 2024 to January 2025). Battery capacity, a
likely value driver, is missing in about 36% of rows, so it is predicted as a
problem of its own before the XGBoost price regressor is trained. DACON
leaderboard normalized RMSE (root mean squared error): **0.919**. The notebook
workflow was later packaged as a command-line pipeline with a seeded holdout
and a median baseline.

## Challenge

Used-EV value is not a simple mileage-and-age rule. Depreciation differs by
manufacturer, model, condition, and drivetrain, and battery capacity is
observed in only 4,786 of 7,497 training rows and 542 of 846 test rows. The
test set has the same gap, so whatever fills the training gap must also work at
prediction time.

## Decisions

**1. Handling missing battery capacity.**
Options: drop the rows, fill one mean or median, add a missing-value flag, or
predict the value from other vehicle fields.
Choice: a separate XGBoost imputer trained only on rows with observed capacity,
using make, model, condition, drivetrain, mileage, and warranty.
Why: in the notebook's ANOVA (analysis of variance) tests, capacity differed by
make, model, condition, and drivetrain (p<0.05) but not by accident history
(p = 0.52), so the other fields carry information a single average throws away;
dropping rows would lose over a third of the data. The CLI fits the imputer on
the development partition only, so validation values do not leak into it.
Cost: imputation errors flow into the price model.

**2. Features that describe depreciation.**
Options: raw columns only, or engineered features for usage and vehicle
segment.
Choice: make x model x condition combinations, mileage per year of age,
odometer km per kWh of capacity, a used flag, a drivetrain weight, and an
accident flag, each created only when its source columns exist.
Why: cars of the same age differ in usage intensity and battery size, and
combinations capture segment-specific depreciation.
Cost: the km-per-kWh feature uses the imputed capacity, so imputation error
enters twice.

**3. Validation.**
Options: none (the competition notebook trained on all rows and relied on the
leaderboard), K-fold, a split grouped by vehicle model, or one seeded holdout.
Choice for the CLI: one seeded 80/20 random holdout plus a
development-partition median baseline, then a refit on all training rows for
the submission.
Why: a quick local check and a reference point without the leaderboard.
Cost: a single random split has unmeasured variance and does not test unseen
vehicle models.

## Results

| Model | Metric | Value | Setting |
|---|---|---:|---|
| XGBoost price model after battery imputation | DACON normalized RMSE | 0.919 | DACON leaderboard |
| Development-partition median price (baseline) | RMSE | - | CLI 80/20 holdout |
| XGBoost price model (CLI) | RMSE | - | CLI 80/20 holdout |

"-": written to `results/metrics.json` when the DACON data is supplied.

## Limitations

- The leaderboard RMSE (0.919) and a local holdout RMSE are not directly comparable; the original data, submission, and leaderboard capture are not included.
- The notebook and the CLI differ (default vs set hyperparameters, test-row vs training-row imputer, no holdout vs holdout); which produced 0.919 is not recorded.
- One random holdout; no time, group, or external validation.
- The gain of model-based imputation over a simple fill was not measured under the same split.
- Drivetrain weights (AWD 1.2, RWD 1.1, FWD 1.0) are hand-set, not learned.

## Next iteration

Keep the submission file, data version, and feature list with every scored
run; compare mean, model-based, and missing-indicator treatments under the
same split; and validate with folds grouped by vehicle model to test
generalisation to models unseen in training.

## Evidence

- [`src/ev_price_prediction_xgb.py`](src/ev_price_prediction_xgb.py)
- [`notebooks/EV_price_prediction_xgb.ipynb`](notebooks/EV_price_prediction_xgb.ipynb)
- [`docs/PROJECT_REVIEW.md`](docs/PROJECT_REVIEW.md)
