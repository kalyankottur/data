#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sktime.forecasting.compose import make_reduction
from sktime.performance_metrics.forecasting import (
    MeanAbsolutePercentageError,
)
from xgboost import XGBRegressor


@dataclass
class Config:
    horizon: int = 7
    cv_window_length: int = 120  # how many unique time points in each training window
    step_length: int = 7
    ar_lags: int = 30  # number of autoregressive lags (days)
    date_features: Optional[List[str]] = None
    target_col: str = "y"
    time_col: str = "ds"
    group_col: Optional[str] = "group"


def build_calendar_features(index: pd.MultiIndex, date_features: Optional[List[str]]) -> pd.DataFrame:
    if not isinstance(index, pd.MultiIndex):
        raise ValueError("Expected a MultiIndex with levels [group, time]")
    if not date_features:
        return pd.DataFrame(index=index)

    time_idx = pd.DatetimeIndex(index.get_level_values(1))

    feats = {}
    for name in date_features:
        if name == "year":
            feats[name] = time_idx.year.astype(int)
        elif name == "month_of_year":
            feats[name] = time_idx.month.astype(int)
        elif name == "day_of_month":
            feats[name] = time_idx.day.astype(int)
        elif name == "day_of_week":
            feats[name] = time_idx.dayofweek.astype(int)
        elif name == "week_of_year":
            feats[name] = time_idx.isocalendar().week.astype(int)
        elif name == "is_weekend":
            feats[name] = (time_idx.dayofweek >= 5).astype(int)
        else:
            raise ValueError(f"Unsupported date feature: {name}")

    X = pd.DataFrame(feats)
    X.index = index
    return X


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = MeanAbsolutePercentageError(symmetric=False)(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def generate_time_splits(unique_times: List[pd.Timestamp], cv_window_length: int, fh: int, step_length: int):
    # slide a window over unique_times; for each fold, train_end is index i-1, test is i..i+fh-1
    splits = []
    if len(unique_times) < cv_window_length + fh:
        return splits
    # start position is cv_window_length
    i = cv_window_length
    while i + fh <= len(unique_times):
        train_end_time = unique_times[i - 1]
        test_times = unique_times[i : i + fh]
        splits.append((train_end_time, test_times))
        i += step_length
    return splits


def fit_global_forecaster(y_train: pd.DataFrame, X_train: Optional[pd.DataFrame], ar_lags: int):
    reg = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=0,
        tree_method="hist",
    )

    forecaster = make_reduction(
        reg,
        strategy="recursive",
        window_length=ar_lags,
        transformers=None,
        pooling="global",
        windows_identical=True,
    )

    forecaster.fit(y_train, X=X_train)
    return forecaster


def sliding_window_cv_global(
    df: pd.DataFrame,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Prepare hierarchical target DataFrame
    grp_col = cfg.group_col or "group"
    df2 = df.copy()
    if cfg.group_col is None or grp_col not in df2.columns:
        df2[grp_col] = "all"

    df2[cfg.time_col] = pd.to_datetime(df2[cfg.time_col])
    df2 = df2.sort_values([grp_col, cfg.time_col])
    # DataFrame with one target column for sktime hierarchical format
    y = df2.set_index([grp_col, cfg.time_col])[[cfg.target_col]]

    # Unique times for global sliding window
    unique_times = sorted(y.index.get_level_values(1).unique())

    splits = generate_time_splits(
        unique_times=unique_times,
        cv_window_length=cfg.cv_window_length,
        fh=cfg.horizon,
        step_length=cfg.step_length,
    )

    fold_rows = []
    per_group_rows = []

    all_true = []
    all_pred = []

    groups = y.index.get_level_values(0).unique()

    for fold_idx, (train_end_time, test_times) in enumerate(splits):
        # Train on all groups up to train_end_time (optionally last cv_window_length times)
        y_train = y.loc[pd.IndexSlice[:, :train_end_time], :]
        # Optionally trim to last cv_window_length unique times to speed up
        train_times = y_train.index.get_level_values(1)
        last_window_times = sorted(train_times.unique())[-cfg.cv_window_length :]
        y_train = y_train.loc[pd.IndexSlice[:, last_window_times], :]

        # Build exogenous calendar features for training
        X_train = build_calendar_features(y_train.index, cfg.date_features)

        forecaster = fit_global_forecaster(
            y_train=y_train,
            X_train=X_train,
            ar_lags=cfg.ar_lags,
        )

        # Predict fh steps ahead for all groups with future exogenous features
        test_times = list(test_times)
        future_index = pd.MultiIndex.from_product(
            [groups, test_times], names=y.index.names
        )
        X_future = build_calendar_features(future_index, cfg.date_features)

        y_pred = forecaster.predict(fh=list(range(1, cfg.horizon + 1)), X=X_future)

        # Collect ground truth for the fold across all groups and horizons
        test_index = future_index.intersection(y.index)
        y_true_fold = y.loc[test_index, :].sort_index()

        # Align predictions with truth
        y_pred_fold = y_pred.reindex(y_true_fold.index)

        # Aggregate metrics for this fold
        fold_metrics = compute_metrics(y_true_fold.values.ravel(), y_pred_fold.values.ravel())
        fold_rows.append({"fold": fold_idx, **fold_metrics})

        all_true.append(y_true_fold.values.ravel())
        all_pred.append(y_pred_fold.values.ravel())

        # Per-group metrics for this fold
        for grp, grp_idx in y_true_fold.groupby(level=0).groups.items():
            yt = y_true_fold.loc[grp]
            yp = y_pred_fold.loc[grp]
            gm = compute_metrics(yt.values.ravel(), yp.values.ravel())
            per_group_rows.append({"fold": fold_idx, grp_col: grp, **gm})

    fold_df = pd.DataFrame(fold_rows)

    # Aggregate across all folds (global)
    if len(all_true) > 0:
        agg_metrics = compute_metrics(np.concatenate(all_true), np.concatenate(all_pred))
    else:
        agg_metrics = {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}

    group_metrics_df = pd.DataFrame(per_group_rows)
    agg_group_metrics = (
        group_metrics_df.groupby(grp_col)[["MAE", "RMSE", "MAPE"]].mean().reset_index()
        if not group_metrics_df.empty
        else pd.DataFrame(columns=[grp_col, "MAE", "RMSE", "MAPE"])
    )

    print("Aggregate metrics across all folds:")
    print(agg_metrics)
    print()
    print("Per-group aggregate metrics (averaged over folds):")
    print(agg_group_metrics)

    return fold_df, agg_group_metrics


def generate_synthetic(n_groups: int = 5, n_days: int = 365, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp("2022-01-01")
    rows = []
    for g in range(n_groups):
        seasonal = np.sin(np.arange(n_days) / 365 * 2 * np.pi) * 10
        trend = np.linspace(0, 5, n_days)
        noise = rng.normal(0, 2.0, n_days)
        level = 50 + g * 5
        values = level + trend + seasonal + noise
        for d in range(n_days):
            rows.append(
                {
                    "group": f"store_{g}",
                    "ds": base_date + pd.Timedelta(days=d),
                    "y": float(values[d]),
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Global XGB with sktime and sliding-window CV")
    parser.add_argument("--csv", type=str, default="", help="Path to CSV with columns: group, ds, y")
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--window", type=int, default=120, help="CV training window length in unique dates")
    parser.add_argument("--step", type=int, default=7)
    parser.add_argument("--lags", type=int, default=30, help="Number of daily autoregressive lags (window_length)")
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        if "group" not in df.columns:
            df["group"] = "all"
    else:
        df = generate_synthetic(n_groups=4, n_days=400)

    cfg = Config(
        horizon=args.horizon,
        cv_window_length=args.window,
        step_length=args.step,
        ar_lags=args.lags,
        date_features=[
            "year",
            "month_of_year",
            "day_of_month",
            "day_of_week",
            "week_of_year",
            "is_weekend",
        ],
        target_col="y",
        time_col="ds",
        group_col="group",
    )

    fold_df, group_metrics_df = sliding_window_cv_global(df, cfg)

    # Save outputs
    fold_df.to_csv("/workspace/cv_fold_metrics.csv", index=False)
    group_metrics_df.to_csv("/workspace/group_metrics.csv", index=False)
    print("Saved metrics to /workspace/cv_fold_metrics.csv and /workspace/group_metrics.csv")


if __name__ == "__main__":
    main()