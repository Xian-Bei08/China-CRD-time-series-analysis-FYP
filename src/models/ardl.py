"""ARDL lag-order selection and fitting."""

from __future__ import annotations

import pandas as pd
from statsmodels.tsa.ardl import ardl_select_order


def prepare_ardl_data(df: pd.DataFrame, y_col: str, x_cols: list[str], year_col: str = "year") -> pd.DataFrame:
    model_df = df[[year_col, y_col] + x_cols].copy().sort_values(year_col).reset_index(drop=True)
    for col in [y_col] + x_cols:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    return model_df.dropna().reset_index(drop=True)


def fit_ardl(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    maxlag: int = 1,
    maxorder: int | dict[str, int] = 1,
    ic: str = "bic",
    trend: str = "c",
    year_col: str = "year",
    causal: bool = False,
):
    model_df = prepare_ardl_data(df, y_col, x_cols, year_col)
    hold_back = maxlag if isinstance(maxorder, dict) else max(maxlag, maxorder)
    if isinstance(maxorder, dict):
        hold_back = max([maxlag] + list(maxorder.values()))

    selector = ardl_select_order(
        endog=model_df[y_col],
        maxlag=maxlag,
        exog=model_df[x_cols],
        maxorder=maxorder,
        ic=ic,
        trend=trend,
        causal=causal,
        hold_back=hold_back,
        missing="drop",
    )
    return model_df, selector, selector.model.fit()
