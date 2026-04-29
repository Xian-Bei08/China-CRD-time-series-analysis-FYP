"""Build clean level and differenced modelling datasets."""

from __future__ import annotations

from src.config import *
from src.loaders import (
    load_crd_data, load_pm25_data, load_ozone_data,
    load_hap_data, load_gdp_ageing_data, load_health_exp_data,
)
from src.data_processing.build_crd import build_crd_main
from src.data_processing.build_macro_vars import build_gdp_ageing_health_main
from src.data_processing.build_risk_factors import build_household_pm, build_ozone, build_pm
from src.data_processing.lag_features import add_trend_feature, build_differenced_modelling_dataset
from src.data_processing.merge_main_dataset import merge_main_dataset


def main() -> None:
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    crd_main = build_crd_main(
        load_crd_data(CRD_FILE), LOCATION, SEX, AGE_NAME, METRIC, CRD_MEASURE, CRD_CAUSE
    )
    pm25_main = build_pm(load_pm25_data(PM25_FILE), LOCATION_ID)
    ozone_main = build_ozone(load_ozone_data(OZONE_FILE), LOCATION_ID)
    household_main = build_household_pm(load_hap_data(HAP_FILE), LOCATION_ID)
    macro_main = build_gdp_ageing_health_main(
        load_gdp_ageing_data(AGEING_FILE),
        load_health_exp_data(GOV_HEALTH_EXP_FILE),
        start_year=START_YEAR,
        end_year=END_YEAR,
        impute_health_exp=True,
    )

    crd_main.to_csv(CRD_MAIN_FILE, index=False)
    pm25_main.to_csv(PM25_MAIN_FILE, index=False)
    ozone_main.to_csv(OZONE_MAIN_FILE, index=False)
    household_main.to_csv(HOUSEHOLD_PM_MAIN_FILE, index=False)
    macro_main.to_csv(AGEING_HEALTH_MAIN_FILE, index=False)

    final_df = merge_main_dataset(
        crd_main=crd_main,
        pm_main=pm25_main,
        ozone_main=ozone_main,
        hap_main=household_main,
        ageing_main=macro_main,
        start_year=START_YEAR,
        end_year=END_YEAR,
    )
    final_df.to_csv(FINAL_ANALYSIS_FILE, index=False)

    add_trend_feature(final_df, YEAR_COL, TREND_COL).to_csv(MODELLING_LEVEL_FILE, index=False)

    diff_df = build_differenced_modelling_dataset(
        df=final_df,
        y_col=LEVEL_Y_COL,
        x_cols=BASE_LEVEL_X_COLS,
        lags=[0, 1, 2],
        year_col=YEAR_COL,
        add_trend=True,
        drop_na=True,
    )
    diff_df.to_csv(MODELLING_DIFF_FILE, index=False)

    print(f"Built datasets: {len(final_df)} level rows, {len(diff_df)} differenced rows.")


if __name__ == "__main__":
    main()
