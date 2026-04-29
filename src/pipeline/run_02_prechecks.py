"""Run concise pre-modelling checks."""

from __future__ import annotations

import pandas as pd

from src.config import (
    BASE_LEVEL_X_COLS, LEVEL_Y_COL, MODELLING_DIFF_FILE, MODELLING_LEVEL_FILE,
    PRECHECK_DIR, SCREENING_CANDIDATE_X_COLS, Y_COL,
)
from src.models.prechecks import (
    check_missing_values,
    descriptive_stats,
    correlation_matrix,
    plot_correlation_heatmap,
    standardize_features,
    calculate_vif,
    stationarity_screen,
    integration_order_screen,
)


def main() -> None:
    tables_dir = PRECHECK_DIR / "tables"
    figures_dir = PRECHECK_DIR / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    diff_df = pd.read_csv(MODELLING_DIFF_FILE)
    level_df = pd.read_csv(MODELLING_LEVEL_FILE)
    diff_cols = [Y_COL] + SCREENING_CANDIDATE_X_COLS

    check_missing_values(diff_df[diff_cols]).to_csv(tables_dir / "missing_diff.csv", index=False)
    descriptive_stats(diff_df, diff_cols).to_csv(tables_dir / "descriptive_diff.csv", index=False)

    corr_df = correlation_matrix(diff_df, diff_cols)
    corr_df.to_csv(tables_dir / "correlation_diff.csv")
    plot_correlation_heatmap(corr_df, figures_dir / "correlation_diff.png")

    vif_input = standardize_features(diff_df, SCREENING_CANDIDATE_X_COLS)
    calculate_vif(vif_input, list(vif_input.columns)).to_csv(tables_dir / "candidate_vif.csv", index=False)

    level_vars = [LEVEL_Y_COL] + BASE_LEVEL_X_COLS
    stationarity_screen(level_df, level_vars, regression="c").to_csv(tables_dir / "stationarity_adf_kpss_za_level.csv", index=False)
    integration_order_screen(level_df, level_vars, regression="c").to_csv(tables_dir / "integration_order_screen.csv", index=False)
    print("Prechecks saved.")


if __name__ == "__main__":
    main()
