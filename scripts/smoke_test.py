"""End-to-end smoke test of the EV-price CLI on small synthetic data.

Writes synthetic train/test CSVs with the DACON column names to a temporary
directory, runs ``src/ev_price_prediction_xgb.py`` on them, and checks that
``metrics.json`` holds the holdout and median-baseline RMSE and that the
submission has one row per test ID. The numbers are synthetic and are not
results.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "src" / "ev_price_prediction_xgb.py"


def synthetic_frame(rows: int, rng: np.random.Generator, start_id: int) -> pd.DataFrame:
    battery = rng.uniform(50, 100, rows)
    battery[rng.random(rows) < 0.2] = np.nan  # exercise the battery imputer
    frame = pd.DataFrame(
        {
            "ID": [f"SMOKE_{start_id + i:04d}" for i in range(rows)],
            "제조사": rng.choice(["A사", "B사", "C사"], rows),
            "모델": rng.choice(["M1", "M2", "M3", "M4"], rows),
            "차량상태": rng.choice(["Brand New", "Nearly New", "Pre-Owned"], rows),
            "배터리용량": battery,
            "구동방식": rng.choice(["AWD", "RWD", "FWD"], rows),
            "주행거리(km)": rng.integers(0, 150_000, rows),
            "보증기간(년)": rng.integers(0, 10, rows),
            "사고이력": rng.choice(["Yes", "No"], rows),
            "연식(년)": rng.integers(0, 3, rows),
        }
    )
    return frame


def main() -> None:
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(prefix="ev_smoke_") as tmp:
        tmp_path = Path(tmp)
        train = synthetic_frame(200, rng, 0)
        train["가격(백만원)"] = (
            30 + 0.4 * train["배터리용량"].fillna(75) - train["주행거리(km)"] / 10_000 + rng.normal(0, 3, len(train))
        )
        test = synthetic_frame(40, rng, 1000)
        train.to_csv(tmp_path / "train.csv", index=False)
        test.to_csv(tmp_path / "test.csv", index=False)

        metrics_path = tmp_path / "metrics.json"
        submission_path = tmp_path / "submission.csv"
        env = {**os.environ, "EV_PRICE_RESULTS_DIR": str(tmp_path / "results")}
        subprocess.run(
            [
                sys.executable, "-B", str(CLI),
                "--train-csv", str(tmp_path / "train.csv"),
                "--test-csv", str(tmp_path / "test.csv"),
                "--output-csv", str(submission_path),
                "--metrics-json", str(metrics_path),
            ],
            check=True,
            cwd=ROOT,
            env=env,
        )

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for key in ("validation_rmse", "baseline_median_rmse"):
            assert key in metrics, f"metrics.json is missing {key}"
            assert math.isfinite(metrics[key]) and metrics[key] >= 0, f"{key} is not a valid RMSE"
        submission = pd.read_csv(submission_path)
        assert list(submission["ID"]) == list(test["ID"]), "submission IDs do not match the test IDs"
        assert submission["가격(백만원)"].notna().all(), "submission has missing predictions"
    print("[smoke] EV-price CLI: OK (metrics.json has holdout and median-baseline RMSE)")


if __name__ == "__main__":
    main()
