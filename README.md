# CRD burden modelling pipeline

This is a cleaned version of the dissertation codebase. It removes old cached files, old result outputs, legacy pipeline fragments, and unused modelling branches.

## Pipeline order

Run from the project root:

```bash
python run_pipeline.py
```

Or run stages individually:

```bash
python -m src.pipeline.run_01_build_modelling_dataset
python -m src.pipeline.run_02_prechecks
python -m src.pipeline.run_03_feature_screening_compare
python -m src.pipeline.run_04_baseline_ols_loocv
python -m src.pipeline.run_05_ardl_ecm
python -m src.pipeline.run_06_diagnostics_and_visuals
```

## Current modelling logic

1. Build level and first-differenced modelling datasets.
2. Run concise prechecks: missingness, descriptive statistics, correlation, VIF, ADF and integration screening.
3. Compare full-sample feature screening paths descriptively.
4. Use nested LOOCV to compare feature-selection paths through OLS prediction performance.
5. Refit the best full-sample selected OLS specification.
6. Map selected differenced variables back to level variables for ARDL-ECM and bounds testing.
7. Generate the CUSUM diagnostic plot for the selected short-run OLS specification.

## Main configuration

All file paths, variable lists, model grids, and ARDL settings are centralised in:

```text
src/config.py
```

