#!/usr/bin/env python
"""Compare LinearRegression and RandomForestRegressor as buildable xorq manifests.

Both models are trained on the same synthetic regression dataset. Each produces a
prediction expression that can be compiled to a YAML manifest via `xorq build`,
then compared with `xorq catalog diff-builds` to see exactly where the two
pipelines diverge.

Usage:
    # Build both manifests
    xorq build model_comparison_build.py -e expr_lr
    xorq build model_comparison_build.py -e expr_rf

    # Register in catalog with aliases
    xorq catalog add builds/<hash_lr> -a model-lr
    xorq catalog add builds/<hash_rf> -a model-rf

    # Diff the two builds
    xorq catalog diff-builds model-lr model-rf --all

    # Run either build to produce predictions
    xorq run builds/<hash_lr> -f json --limit 5
    xorq run builds/<hash_rf> -f json --limit 5
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

import xorq.api as xo
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline

# ---------------------------------------------------------------------------
# 1. Synthetic regression dataset (deterministic)
# ---------------------------------------------------------------------------
target = "target"
feature_prefix = "feature_"

X, y = make_regression(
    n_samples=2000,
    n_features=10,
    n_informative=6,
    noise=10.0,
    random_state=42,
)

con = xo.connect()
df = (
    pd.DataFrame(X)
    .rename(columns=(feature_prefix + "{}").format)
    .assign(**{target: y})
)
data = con.register(df, "regression_data")

# ---------------------------------------------------------------------------
# 2. Train/test split (expression-level, captured in the manifest)
# ---------------------------------------------------------------------------
train, test = xo.train_test_splits(
    data,
    test_sizes=0.2,
    num_buckets=10000,
    random_seed=42,
)

features = data.select(xo.selectors.startswith(feature_prefix)).columns

# ---------------------------------------------------------------------------
# 3. Two sklearn pipelines
# ---------------------------------------------------------------------------
lr_sklearn = SkPipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression()),
])

rf_sklearn = SkPipeline([
    ("scaler", StandardScaler()),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
    )),
])

# ---------------------------------------------------------------------------
# 4. Fit and predict with xorq Pipeline (deferred expressions)
# ---------------------------------------------------------------------------
lr_fitted = Pipeline.from_instance(lr_sklearn).fit(
    train, features=features, target=target
)
expr_lr = lr_fitted.predict(test)

rf_fitted = Pipeline.from_instance(rf_sklearn).fit(
    train, features=features, target=target
)
expr_rf = rf_fitted.predict(test)

# ---------------------------------------------------------------------------
# 5. Shared regression metrics (computed at runtime, not in the build)
# ---------------------------------------------------------------------------
metric_configs = (
    ("mae", mean_absolute_error, {}),
    ("mse", mean_squared_error, {}),
    ("r2", r2_score, {}),
)


def compute_deferred_metrics(predictions_expr):
    return {
        name: deferred_sklearn_metric(
            expr=predictions_expr,
            target=target,
            pred_col="predicted",
            metric_fn=metric_fn,
            metric_kwargs=kwargs if kwargs else (),
        )
        for name, metric_fn, kwargs in metric_configs
    }


if __name__ == "__pytest_main__":
    lr_metrics = compute_deferred_metrics(expr_lr)
    rf_metrics = compute_deferred_metrics(expr_rf)

    print("=== Linear Regression ===")
    for name, metric_expr in lr_metrics.items():
        print(f"  {name}: {metric_expr.execute():.4f}")

    print("\n=== Random Forest Regressor ===")
    for name, metric_expr in rf_metrics.items():
        print(f"  {name}: {metric_expr.execute():.4f}")

    lr_df = expr_lr.execute()
    rf_df = expr_rf.execute()
    assert "predicted" in lr_df.columns
    assert "predicted" in rf_df.columns
    assert len(lr_df) == len(rf_df)

    pytest_examples_passed = True
