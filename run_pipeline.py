"""Run the full dissertation modelling pipeline in order."""

from src.pipeline import (
    run_01_build_modelling_dataset,
    run_02_prechecks,
    run_03_baseline_ols_loocv,
    run_04_ardl_ecm,
    run_05_diagnostics_and_visuals,
)


def main() -> None:
    steps = [
        run_01_build_modelling_dataset.main,
        run_02_prechecks.main,
        run_03_baseline_ols_loocv.main,
        run_04_ardl_ecm.main,
        run_05_diagnostics_and_visuals.main,
    ]
    for step in steps:
        step()


if __name__ == "__main__":
    main()
