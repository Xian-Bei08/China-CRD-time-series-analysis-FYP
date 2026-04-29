"""Nested-LOOCV helpers for OLS model comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.models.feature_selection import (
    backward_elimination,
    elastic_net_select_train_only,
    random_forest_importance,
    vif_screening_path,
)


def prepare_validation_data(df: pd.DataFrame, y_col: str, x_cols: list[str], year_col: str = "year") -> pd.DataFrame:
    model_df = df[[year_col, y_col] + x_cols].copy().sort_values(year_col).reset_index(drop=True)
    for col in [y_col] + x_cols:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    return model_df.dropna(subset=[y_col] + x_cols).reset_index(drop=True)


def _select_features_for_path(
    train_df: pd.DataFrame,
    y_col: str,
    candidate_cols: list[str],
    path_name: str,
    *,
    alpha_grid: list[float],
    l1_grid: list[float],
    inner_cv_splits: int,
    random_state: int,
    backward_pvalue_threshold: float,
    random_forest_n_estimators: int,
    vif_threshold: float,
) -> tuple[list[str], dict]:
    if path_name == "elastic_net":
        return elastic_net_select_train_only(
            train_df, y_col, candidate_cols, alpha_grid, l1_grid, inner_cv_splits, random_state
        )
    if path_name == "backward_elimination":
        path_df, selected = backward_elimination(train_df, y_col, candidate_cols, backward_pvalue_threshold)
        return selected, {"n_steps": len(path_df)}
    if path_name == "random_forest":
        imp_df, selected = random_forest_importance(
            train_df, y_col, candidate_cols, random_forest_n_estimators, random_state
        )
        return selected, {"top_variable": str(imp_df.iloc[0]["variable"])}
    if path_name == "vif_filtering":
        path_df, selected = vif_screening_path(train_df, candidate_cols, vif_threshold)
        return selected, {"n_steps": len(path_df)}
    raise ValueError(f"Unknown path name: {path_name}")


def nested_loocv_single_path(
    df: pd.DataFrame,
    y_col: str,
    candidate_cols: list[str],
    path_name: str,
    year_col: str = "year",
    *,
    alpha_grid: list[float],
    l1_grid: list[float],
    inner_cv_splits: int,
    random_state: int,
    backward_pvalue_threshold: float,
    random_forest_n_estimators: int,
    vif_threshold: float,
) -> pd.DataFrame:
    model_df = prepare_validation_data(df, y_col, candidate_cols, year_col)
    rows = []

    for i in range(len(model_df)):
        train_df = model_df.drop(index=i).reset_index(drop=True)
        test_df = model_df.iloc[[i]].reset_index(drop=True)

        selected, meta = _select_features_for_path(
            train_df=train_df,
            y_col=y_col,
            candidate_cols=candidate_cols,
            path_name=path_name,
            alpha_grid=alpha_grid,
            l1_grid=l1_grid,
            inner_cv_splits=inner_cv_splits,
            random_state=random_state,
            backward_pvalue_threshold=backward_pvalue_threshold,
            random_forest_n_estimators=random_forest_n_estimators,
            vif_threshold=vif_threshold,
        )

        train_clean = prepare_validation_data(train_df, y_col, selected, year_col)
        test_clean = prepare_validation_data(test_df, y_col, selected, year_col)
        model = sm.OLS(
            train_clean[y_col],
            sm.add_constant(train_clean[selected], has_constant="add"),
        ).fit()
        pred = float(model.predict(sm.add_constant(test_clean[selected], has_constant="add")).iloc[0])
        actual = float(test_clean[y_col].iloc[0])

        row = {
            "path": path_name,
            year_col: int(test_clean[year_col].iloc[0]),
            "train_size": len(train_clean),
            "n_selected": len(selected),
            "selected_variables": " | ".join(selected),
            "actual": actual,
            "predicted": pred,
            "error": actual - pred,
            "abs_error": abs(actual - pred),
            "squared_error": (actual - pred) ** 2,
            "ape": abs((actual - pred) / actual) * 100 if actual != 0 else np.nan,
        }
        row.update({f"meta_{k}": v for k, v in meta.items()})
        rows.append(row)

    return pd.DataFrame(rows)


def metrics_table(validation_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    return pd.DataFrame({
        "model": [model_name],
        "n_forecasts": [len(validation_df)],
        "mean_error": [validation_df["error"].mean()],
        "mae": [validation_df["abs_error"].mean()],
        "rmse": [float(np.sqrt(validation_df["squared_error"].mean()))],
        "mape_percent": [validation_df["ape"].mean()],
        "directional_accuracy_percent": [float((np.sign(validation_df["actual"]) == np.sign(validation_df["predicted"])).mean() * 100)],
    })


def selection_frequency_table(detail_df: pd.DataFrame, path_name: str) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for selected in detail_df["selected_variables"].fillna(""):
        for var in [v.strip() for v in str(selected).split("|") if v.strip()]:
            counts[var] = counts.get(var, 0) + 1
    rows = [
        {"path": path_name, "variable": var, "selected_in_folds": n, "selection_rate": n / len(detail_df)}
        for var, n in counts.items()
    ]
    return pd.DataFrame(rows).sort_values(["selected_in_folds", "variable"], ascending=[False, True]) if rows else pd.DataFrame()


def compare_nested_loocv_paths(
    df: pd.DataFrame,
    y_col: str,
    candidate_cols: list[str],
    path_names: list[str],
    year_col: str = "year",
    *,
    alpha_grid: list[float],
    l1_grid: list[float],
    inner_cv_splits: int,
    random_state: int,
    backward_pvalue_threshold: float,
    random_forest_n_estimators: int,
    vif_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    details, metrics, freqs = [], [], []
    for path in path_names:
        path_details = nested_loocv_single_path(
            df, y_col, candidate_cols, path, year_col,
            alpha_grid=alpha_grid,
            l1_grid=l1_grid,
            inner_cv_splits=inner_cv_splits,
            random_state=random_state,
            backward_pvalue_threshold=backward_pvalue_threshold,
            random_forest_n_estimators=random_forest_n_estimators,
            vif_threshold=vif_threshold,
        )
        path_metrics = metrics_table(path_details, path)
        path_metrics["mean_n_selected"] = float(path_details["n_selected"].mean())
        details.append(path_details)
        metrics.append(path_metrics)
        freqs.append(selection_frequency_table(path_details, path))

    return (
        pd.concat(details, ignore_index=True),
        pd.concat(metrics, ignore_index=True).sort_values("rmse").reset_index(drop=True),
        pd.concat(freqs, ignore_index=True),
    )
