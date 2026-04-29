"""Produce ARDL residual diagnostics and residual autocorrelation plots."""

from __future__ import annotations

import pandas as pd

from src.config import ARDL_IC, ARDL_MAXLAG, ARDL_MAXORDER, ARDL_TREND, DIAGNOSTICS_DIR, LEVEL_Y_COL, MODELLING_LEVEL_FILE
from src.models.ardl import fit_ardl
from src.models.diagnostics import ardl_residual_diagnostics, plot_residual_acf_pacf
from src.models.selection_results import load_final_selected_variables, map_diff_to_level_variables


def main() -> None:
    tables_dir = DIAGNOSTICS_DIR / "tables"
    figures_dir = DIAGNOSTICS_DIR / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    level_df = pd.read_csv(MODELLING_LEVEL_FILE)
    selected_diff_vars = load_final_selected_variables()
    ardl_x_cols = map_diff_to_level_variables(selected_diff_vars)
    ardl_maxorder = {col: ARDL_MAXORDER[col] for col in ardl_x_cols}

    _, _, ardl_res = fit_ardl(
        df=level_df,
        y_col=LEVEL_Y_COL,
        x_cols=ardl_x_cols,
        maxlag=ARDL_MAXLAG,
        maxorder=ardl_maxorder,
        ic=ARDL_IC,
        trend=ARDL_TREND,
    )

    ardl_residual_diagnostics(ardl_res).to_csv(tables_dir / "ardl_diagnostics.csv", index=False)
    plot_residual_acf_pacf(ardl_res, figures_dir / "ardl_residual_acf_pacf.png")


if __name__ == "__main__":
    main()
