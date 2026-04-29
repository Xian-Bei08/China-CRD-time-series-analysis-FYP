"""Fit ARDL-ECM using the selected variables from the best LOOCV path."""

from __future__ import annotations

import pandas as pd

from src.config import (
    ARDL_IC,
    ARDL_MAXLAG,
    ARDL_MAXORDER,
    ARDL_TREND,
    BOUNDS_CASE,
    BOUNDS_ECM_DIR,
    LEVEL_Y_COL,
    MODELLING_LEVEL_FILE,
    YEAR_COL,
)
from src.models.bounds_ecm import (
    bounds_interpretation_text,
    bounds_result_table,
    ci_table,
    ecm_speed_of_adjustment_table,
    fitted_differences_export,
    fit_ardl_then_bounds,
    plot_uecm_actual_vs_fitted,
    save_text,
    uecm_coefficients_table,
)
from src.models.selection_results import (
    load_best_path,
    load_final_selected_variables,
    map_diff_to_level_variables,
)


def main() -> None:
    tables_dir = BOUNDS_ECM_DIR / "tables"
    figures_dir = BOUNDS_ECM_DIR / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    level_df = pd.read_csv(MODELLING_LEVEL_FILE)
    best_path = load_best_path()
    selected_diff_vars = load_final_selected_variables()
    ardl_x_cols = map_diff_to_level_variables(selected_diff_vars)
    ardl_maxorder = {col: ARDL_MAXORDER[col] for col in ardl_x_cols}

    results = fit_ardl_then_bounds(
        df=level_df,
        y_col=LEVEL_Y_COL,
        x_cols=ardl_x_cols,
        maxlag=ARDL_MAXLAG,
        maxorder=ardl_maxorder,
        ic=ARDL_IC,
        trend=ARDL_TREND,
        year_col=YEAR_COL,
        causal=False,
        bounds_case=BOUNDS_CASE,
    )

    ardl_res = results["ardl_res"]
    uecm_res = results["uecm_res"]
    bounds_res = results["bounds_res"]

    bounds_result_table(bounds_res).to_csv(tables_dir / "bounds_test_results.csv", index=False)
    bounds_res.crit_vals.to_csv(tables_dir / "bounds_test_critical_values.csv")
    uecm_coefficients_table(uecm_res).to_csv(tables_dir / "uecm_coefficients.csv", index=False)
    ecm_speed_of_adjustment_table(uecm_res).to_csv(tables_dir / "ecm_speed_of_adjustment.csv", index=False)

    fitted = fitted_differences_export(results["model_df"], uecm_res, LEVEL_Y_COL, YEAR_COL)
    fitted.to_csv(tables_dir / "uecm_actual_vs_fitted.csv", index=False)
    plot_uecm_actual_vs_fitted(fitted, LEVEL_Y_COL, figures_dir / "uecm_actual_vs_fitted.png", YEAR_COL)

    save_text(ardl_res.summary().as_text(), tables_dir / "ardl_summary.txt")
    save_text(uecm_res.summary().as_text(), tables_dir / "uecm_summary.txt")
    save_text(str(bounds_res), tables_dir / "bounds_test.txt")
    save_text(bounds_interpretation_text(bounds_res, case=BOUNDS_CASE), tables_dir / "bounds_interpretation.txt")
    save_text(str(results["selector"].model.ardl_order), tables_dir / "ardl_selected_order.txt")
    save_text(
        f"best_path={best_path}\nselected_diff_vars={selected_diff_vars}\nardl_level_vars={ardl_x_cols}\n",
        tables_dir / "model_selection_notes.txt",
    )

    try:
        ci_table(uecm_res).to_csv(tables_dir / "cointegrating_vector.csv", index=False)
        save_text(str(uecm_res.ci_summary()), tables_dir / "cointegrating_vector_summary.txt")
    except Exception as exc:
        save_text(f"Cointegrating vector unavailable: {exc}\n", tables_dir / "cointegrating_vector_summary.txt")


if __name__ == "__main__":
    main()
