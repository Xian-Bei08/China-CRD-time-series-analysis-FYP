"""Compare feature-selection paths using nested LOOCV and refit the best OLS model."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from src.config import (
    BACKWARD_PVALUE_THRESHOLD, ELASTIC_NET_ALPHA_GRID, ELASTIC_NET_L1_GRID,
    FEATURE_SELECTION_PATHS, INNER_CV_SPLITS, MODELLING_DIFF_FILE, OLS_LOOCV_DIR,
    RANDOM_FOREST_N_ESTIMATORS, RANDOM_STATE, SCREENING_CANDIDATE_X_COLS,
    VIF_THRESHOLD, YEAR_COL, Y_COL,
)
from src.models.feature_selection import (
    backward_elimination, elastic_net_select_train_only, random_forest_importance,
    vif_screening_path,
)
from src.models.validation import compare_nested_loocv_paths


def final_selection(df: pd.DataFrame, path: str) -> tuple[list[str], dict]:
    if path == "elastic_net":
        return elastic_net_select_train_only(df, Y_COL, SCREENING_CANDIDATE_X_COLS, ELASTIC_NET_ALPHA_GRID, ELASTIC_NET_L1_GRID, INNER_CV_SPLITS, RANDOM_STATE)
    if path == "backward_elimination":
        path_df, selected = backward_elimination(df, Y_COL, SCREENING_CANDIDATE_X_COLS, BACKWARD_PVALUE_THRESHOLD)
        return selected, {"n_steps": len(path_df)}
    if path == "random_forest":
        imp_df, selected = random_forest_importance(df, Y_COL, SCREENING_CANDIDATE_X_COLS, RANDOM_FOREST_N_ESTIMATORS, RANDOM_STATE)
        return selected, {"top_variable": imp_df.iloc[0]["variable"]}
    if path == "vif_filtering":
        path_df, selected = vif_screening_path(df, SCREENING_CANDIDATE_X_COLS, VIF_THRESHOLD)
        return selected, {"n_steps": len(path_df)}
    raise ValueError(f"Unknown selection path: {path}")


def main() -> None:
    tables_dir = OLS_LOOCV_DIR / "tables"
    figures_dir = OLS_LOOCV_DIR / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MODELLING_DIFF_FILE)
    details, metrics, frequencies = compare_nested_loocv_paths(
        df=df,
        y_col=Y_COL,
        candidate_cols=SCREENING_CANDIDATE_X_COLS,
        path_names=FEATURE_SELECTION_PATHS,
        year_col=YEAR_COL,
        alpha_grid=ELASTIC_NET_ALPHA_GRID,
        l1_grid=ELASTIC_NET_L1_GRID,
        inner_cv_splits=INNER_CV_SPLITS,
        random_state=RANDOM_STATE,
        backward_pvalue_threshold=BACKWARD_PVALUE_THRESHOLD,
        random_forest_n_estimators=RANDOM_FOREST_N_ESTIMATORS,
        vif_threshold=VIF_THRESHOLD,
    )
    details.to_csv(tables_dir / "nested_loocv_prediction_paths.csv", index=False)
    metrics.to_csv(tables_dir / "nested_loocv_metrics_summary.csv", index=False)
    frequencies.to_csv(tables_dir / "nested_loocv_selection_frequency.csv", index=False)

    best_path = str(metrics.iloc[0]["model"])
    best_x_cols, meta = final_selection(df, best_path)
    pd.DataFrame([{
        "best_path": best_path,
        "final_selected_variables": " | ".join(best_x_cols),
        "n_selected": len(best_x_cols),
        **{f"meta_{k}": v for k, v in meta.items()},
    }]).to_csv(tables_dir / "best_path_final_selection.csv", index=False)

    model_df = df[[YEAR_COL, Y_COL] + best_x_cols].dropna().reset_index(drop=True)
    ols = sm.OLS(model_df[Y_COL], sm.add_constant(model_df[best_x_cols], has_constant="add")).fit()
    pd.DataFrame({
        "variable": ols.params.index,
        "coefficient": ols.params.values,
        "std_error": ols.bse.values,
        "t_value": ols.tvalues.values,
        "p_value": ols.pvalues.values,
    }).to_csv(tables_dir / "best_path_ols_coefficients.csv", index=False)
    (tables_dir / "best_path_ols_summary.txt").write_text(ols.summary().as_text(), encoding="utf-8")

    best_details = details[details["path"] == best_path]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(best_details[YEAR_COL], best_details["actual"], marker="o", label="Actual")
    ax.plot(best_details[YEAR_COL], best_details["predicted"], marker="o", label="Predicted")
    ax.set_title(f"Nested LOOCV Actual vs Predicted: {best_path}")
    ax.set_ylabel(Y_COL)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "loocv_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(best_details[YEAR_COL], best_details["error"], marker="o")
    ax.axhline(0, linestyle="--")
    ax.set_title(f"Nested LOOCV Prediction Error: {best_path}")
    ax.set_ylabel("Prediction error")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "loocv_prediction_error_trend.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Best path: {best_path}; selected variables: {best_x_cols}")


if __name__ == "__main__":
    main()
