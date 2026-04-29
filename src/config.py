"""Central configuration for the dissertation modelling pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

CRD_FILE = DATA_RAW / "MainDependentVariable.csv"
PM25_FILE = DATA_RAW / "AIR_POLLUTION_1990_2023_PM.CSV"
OZONE_FILE = DATA_RAW / "AIR_POLLUTION_1990_2021_OZONE.csv"
HAP_FILE = DATA_RAW / "AIR_POLLUTION_1990_2023_HAP_PM.CSV"
AGEING_FILE = DATA_RAW / "fuels_gdp_ageing.csv"
GOV_HEALTH_EXP_FILE = DATA_RAW / "Domestic general government health expenditure.csv"

CRD_MAIN_FILE = DATA_INTERIM / "crd_main.csv"
PM25_MAIN_FILE = DATA_INTERIM / "pm25_main.csv"
OZONE_MAIN_FILE = DATA_INTERIM / "ozone_main.csv"
HOUSEHOLD_PM_MAIN_FILE = DATA_INTERIM / "household_pm_main.csv"
AGEING_HEALTH_MAIN_FILE = DATA_INTERIM / "ageing_health_main.csv"

FINAL_ANALYSIS_FILE = DATA_PROCESSED / "final_analysis_dataset.csv"
MODELLING_DIFF_FILE = DATA_PROCESSED / "modelling_dataset_diff.csv"
MODELLING_LEVEL_FILE = DATA_PROCESSED / "modelling_dataset_level.csv"

PRECHECK_DIR = RESULTS_DIR / "prechecks"
OLS_LOOCV_DIR = RESULTS_DIR / "ols_loocv"
BOUNDS_ECM_DIR = RESULTS_DIR / "bounds_ecm"
DIAGNOSTICS_DIR = RESULTS_DIR / "diagnostics"

LOCATION = "China"
LOCATION_ID = 6
SEX = "Both"
AGE_NAME = "Age-standardized"
METRIC = "Rate"
CRD_MEASURE = "DALYs (Disability-Adjusted Life Years)"
CRD_CAUSE = "Chronic respiratory diseases"

START_YEAR = 1990
END_YEAR = 2023
YEAR_COL = "year"
LEVEL_Y_COL = "crd_daly_rate"
Y_COL = "d_crd_daly_rate"
TREND_COL = "trend"

BASE_LEVEL_X_COLS = [
    "pm25",
    "ozone",
    "household_pm",
    "ageing_65_plus",
    "gov_health_exp_pct_gdp",
]

SCREENING_CANDIDATE_X_COLS = [
    "d_pm25_lag0",
    "d_pm25_lag1",
    "d_pm25_lag2",
    "d_ozone_lag0",
    "d_ozone_lag1",
    "d_ozone_lag2",
    "d_household_pm_lag0",
    "d_household_pm_lag1",
    "d_household_pm_lag2",
    "d_ageing_65_plus_lag0",
    "d_ageing_65_plus_lag1",
    "d_ageing_65_plus_lag2",
    "d_gov_health_exp_pct_gdp_lag0",
    "d_gov_health_exp_pct_gdp_lag1",
    "d_gov_health_exp_pct_gdp_lag2",
    "trend",
]

FEATURE_SELECTION_PATHS = [
    "elastic_net",
    "backward_elimination",
    "random_forest",
    "vif_filtering",
]

ELASTIC_NET_ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
ELASTIC_NET_L1_GRID = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
BACKWARD_PVALUE_THRESHOLD = 0.10
RANDOM_FOREST_N_ESTIMATORS = 500
VIF_THRESHOLD = 10.0
INNER_CV_SPLITS = 5
RANDOM_STATE = 42

ARDL_MAXLAG = 1
ARDL_MAXORDER = {
    "household_pm": 1,
    "pm25": 0,
    "ozone": 0,
    "ageing_65_plus": 1,
    "gov_health_exp_pct_gdp": 0,
    "trend": 0,
}
ARDL_IC = "bic"
ARDL_TREND = "c"
BOUNDS_CASE = 3
