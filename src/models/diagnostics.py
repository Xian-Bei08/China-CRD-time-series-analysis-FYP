"""Diagnostic utilities for the ARDL model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import jarque_bera


def ardl_residual_diagnostics(model, bg_lags: int = 2) -> pd.DataFrame:
    resid = pd.Series(model.resid).dropna().reset_index(drop=True)
    fitted = pd.Series(model.fittedvalues).dropna().reset_index(drop=True)

    n = min(len(resid), len(fitted))
    resid = resid.iloc[-n:].reset_index(drop=True)
    fitted = fitted.iloc[-n:].reset_index(drop=True)

    bg_df = pd.DataFrame({"resid": resid, "fitted": fitted})
    for lag in range(1, bg_lags + 1):
        bg_df[f"resid_lag{lag}"] = bg_df["resid"].shift(lag)

    bg_df = bg_df.dropna()
    bg_y = bg_df["resid"]
    bg_x = sm.add_constant(bg_df.drop(columns="resid"), has_constant="add")
    bg_model = sm.OLS(bg_y, bg_x).fit()

    bg_stat = len(bg_df) * bg_model.rsquared
    bg_pvalue = chi2.sf(bg_stat, bg_lags)

    white_x = sm.add_constant(pd.DataFrame({"fitted": fitted}), has_constant="add")
    white_stat, white_pvalue, _, _ = het_white(resid, white_x)

    jb_stat, jb_pvalue, _, _ = jarque_bera(resid)

    return pd.DataFrame(
        {
            "test": [
                "Breusch-Godfrey serial correlation",
                "White heteroskedasticity",
                "Jarque-Bera normality",
            ],
            "statistic": [bg_stat, white_stat, jb_stat],
            "p_value": [bg_pvalue, white_pvalue, jb_pvalue],
        }
    )


def plot_residual_acf_pacf(model, output_file: Path) -> None:
    resid = pd.Series(model.resid).dropna()

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    plot_acf(resid, ax=axes[0])
    plot_pacf(resid, ax=axes[1], method="ywm")
    axes[0].set_title("Residual ACF")
    axes[1].set_title("Residual PACF")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
