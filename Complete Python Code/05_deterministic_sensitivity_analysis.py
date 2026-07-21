#!/usr/bin/env python3
"""
05_deterministic_sensitivity_analysis.py

Purpose
-------
This script performs deterministic sensitivity analysis across all biologically
plausible candidate parameter sets from the deterministic parameter-screening step.

It re-runs the deterministic population model for every candidate parameter set,
then summarizes how often clustered low-exposure troughs produce a higher final
high-MIC fraction than dispersed troughs.

Required input file
-------------------
Place ONE of the following in the same folder as this script:

1) parameter_screening_candidate_results.csv
   - preferred
   - contains only biologically plausible candidate parameter sets

OR

2) parameter_screening_all_results.csv
   - if the candidate file is not available
   - the script will keep only rows where is_candidate == True

Outputs
-------
All outputs are saved to:

    deterministic_sensitivity_output/

Main output files:
    deterministic_sensitivity_per_parameter_species.csv
    deterministic_sensitivity_species_summary.csv
    deterministic_sensitivity_overall_summary.csv
    figure_sensitivity_percent_clustered_gt_dispersed.png
    figure_sensitivity_median_clustered_minus_dispersed.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. File paths
# ============================================================

DATA_DIR = Path(".")
CANDIDATE_FILE = DATA_DIR / "parameter_screening_candidate_results.csv"
ALL_RESULTS_FILE = DATA_DIR / "parameter_screening_all_results.csv"

OUTPUT_DIR = DATA_DIR / "deterministic_sensitivity_output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Representative MIC phenotypes from Table 3
# ============================================================

REPRESENTATIVE_MIC = {
    "Escherichia coli": {
        "short_name": "E. coli",
        "low_mic": 0.016,
        "high_mic": 0.125,
    },
    "Klebsiella pneumoniae": {
        "short_name": "K. pneumoniae",
        "low_mic": 0.03,
        "high_mic": 0.25,
    },
    "Pseudomonas aeruginosa": {
        "short_name": "P. aeruginosa",
        "low_mic": 0.125,
        "high_mic": 1.0,
    },
}


# ============================================================
# 3. Fixed simulation settings
# ============================================================

TOTAL_TIME = 72.0
DT = 0.1
DOSE_TIMES = [0, 12, 24, 36, 48, 60]
SCENARIOS = ["regular", "dispersed_troughs", "clustered_troughs"]

INITIAL_TOTAL_POPULATION = 1e6
INITIAL_HIGH_FRACTION = 1e-4
CARRYING_CAPACITY = 1e9
HIGH_MIC_FITNESS_COST = 0.05
NUMERICAL_FLOOR = 1e-300
EXTINCTION_THRESHOLD = 1e3


# ============================================================
# 4. Helper functions: input loading
# ============================================================

def load_candidate_parameter_sets() -> pd.DataFrame:
    """
    Load biologically plausible candidate parameter sets.

    Preferred input:
        parameter_screening_candidate_results.csv

    Fallback:
        parameter_screening_all_results.csv filtered to is_candidate == True
    """
    if CANDIDATE_FILE.exists():
        df = pd.read_csv(CANDIDATE_FILE)
        source = str(CANDIDATE_FILE)
    elif ALL_RESULTS_FILE.exists():
        df = pd.read_csv(ALL_RESULTS_FILE)
        if "is_candidate" not in df.columns:
            raise ValueError(
                "parameter_screening_all_results.csv does not contain 'is_candidate'."
            )
        df = df[df["is_candidate"] == True].copy()
        source = str(ALL_RESULTS_FILE)
    else:
        raise FileNotFoundError(
            "No input file found.\n"
            "Place 'parameter_screening_candidate_results.csv' or "
            "'parameter_screening_all_results.csv' in the same folder as this script."
        )

    if len(df) == 0:
        raise ValueError(
            "No candidate parameter sets were found. "
            "Check the parameter-screening output files."
        )

    required_columns = [
        "full_dose",
        "reduced_dose_fraction",
        "reduced_dose",
        "half_life",
        "baseline_growth_rate",
        "max_drug_effect",
        "ec50_ratio",
        "hill_coefficient",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required parameter columns: {missing}"
        )

    print(f"Loaded candidate parameter sets from: {source}")
    print(f"Number of candidate parameter sets: {len(df)}")

    return df.reset_index(drop=True)


# ============================================================
# 5. Exposure trajectory generation
# ============================================================

def generate_time_points() -> np.ndarray:
    return np.round(np.arange(0, TOTAL_TIME + DT, DT), 6)


def build_dose_patterns(full_dose: float, reduced_dose: float) -> dict:
    return {
        "regular": [
            full_dose, full_dose, full_dose, full_dose, full_dose, full_dose
        ],
        "dispersed_troughs": [
            full_dose, reduced_dose, full_dose, full_dose, reduced_dose, full_dose
        ],
        "clustered_troughs": [
            full_dose, full_dose, reduced_dose, reduced_dose, full_dose, full_dose
        ],
    }


def generate_concentration_trajectory(dose_pattern: list, half_life: float) -> np.ndarray:
    time_points = generate_time_points()
    dose_dict = dict(zip(DOSE_TIMES, dose_pattern))
    decay_constant = np.log(2) / half_life

    concentrations = []
    concentration = 0.0

    for t in time_points:
        if float(t) in dose_dict:
            concentration += dose_dict[float(t)]

        concentrations.append(concentration)
        concentration = concentration * np.exp(-decay_constant * DT)

    return np.array(concentrations)


def generate_exposure_trajectories(
    full_dose: float,
    reduced_dose_fraction: float,
    half_life: float,
) -> pd.DataFrame:
    reduced_dose = full_dose * reduced_dose_fraction
    dose_patterns = build_dose_patterns(full_dose, reduced_dose)

    trajectory_df = pd.DataFrame({"time_h": generate_time_points()})

    for scenario, pattern in dose_patterns.items():
        trajectory_df[scenario] = generate_concentration_trajectory(pattern, half_life)

    return trajectory_df


# ============================================================
# 6. Deterministic population model
# ============================================================

def drug_effect(
    concentration: float,
    mic: float,
    max_drug_effect: float,
    ec50_ratio: float,
    hill_coefficient: float,
) -> float:
    exposure_ratio = concentration / mic
    numerator = exposure_ratio ** hill_coefficient
    denominator = (ec50_ratio ** hill_coefficient) + numerator
    if denominator <= 0:
        return 0.0
    return max_drug_effect * numerator / denominator


def simulate_one_condition(
    trajectory_df: pd.DataFrame,
    species: str,
    scenario: str,
    low_mic: float,
    high_mic: float,
    baseline_growth_rate: float,
    max_drug_effect: float,
    ec50_ratio: float,
    hill_coefficient: float,
) -> pd.DataFrame:
    time_values = trajectory_df["time_h"].to_numpy()
    concentration_values = trajectory_df[scenario].to_numpy()
    dt = float(np.median(np.diff(time_values)))

    n_high = INITIAL_TOTAL_POPULATION * INITIAL_HIGH_FRACTION
    n_low = INITIAL_TOTAL_POPULATION - n_high

    rows = []

    for idx, time_h in enumerate(time_values):
        concentration = float(concentration_values[idx])
        n_total = n_low + n_high
        high_fraction = n_high / n_total if n_total > 0 else np.nan

        rows.append({
            "time_h": time_h,
            "species": species,
            "scenario": scenario,
            "concentration_mg_l": concentration,
            "low_mic": low_mic,
            "high_mic": high_mic,
            "N_low": n_low,
            "N_high": n_high,
            "N_total": n_total,
            "high_fraction": high_fraction,
        })

        if idx == len(time_values) - 1:
            break

        logistic_factor = max(0.0, 1.0 - (n_total / CARRYING_CAPACITY))
        growth_low = baseline_growth_rate * logistic_factor
        growth_high = baseline_growth_rate * (1.0 - HIGH_MIC_FITNESS_COST) * logistic_factor

        effect_low = drug_effect(
            concentration, low_mic, max_drug_effect, ec50_ratio, hill_coefficient
        )
        effect_high = drug_effect(
            concentration, high_mic, max_drug_effect, ec50_ratio, hill_coefficient
        )

        net_growth_low = growth_low - effect_low
        net_growth_high = growth_high - effect_high

        n_low = n_low * np.exp(net_growth_low * dt)
        n_high = n_high * np.exp(net_growth_high * dt)

        n_low = max(n_low, NUMERICAL_FLOOR)
        n_high = max(n_high, NUMERICAL_FLOOR)

    return pd.DataFrame(rows)


def simulate_all_conditions(trajectory_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    all_ts = []

    for species, mic_info in REPRESENTATIVE_MIC.items():
        for scenario in SCENARIOS:
            ts = simulate_one_condition(
                trajectory_df=trajectory_df,
                species=species,
                scenario=scenario,
                low_mic=mic_info["low_mic"],
                high_mic=mic_info["high_mic"],
                baseline_growth_rate=params["baseline_growth_rate"],
                max_drug_effect=params["max_drug_effect"],
                ec50_ratio=params["ec50_ratio"],
                hill_coefficient=params["hill_coefficient"],
            )
            all_ts.append(ts)

    return pd.concat(all_ts, ignore_index=True)


# ============================================================
# 7. Sensitivity analysis metrics
# ============================================================

def summarize_candidate_by_species(
    timeseries_df: pd.DataFrame,
    parameter_id: int,
    params: dict,
) -> pd.DataFrame:
    """
    For one candidate parameter set, compute final outcome metrics for each species.
    """
    rows = []

    for species in REPRESENTATIVE_MIC.keys():
        species_df = timeseries_df[timeseries_df["species"] == species].copy()

        # Extract final rows for each scenario
        final_rows = {}
        for scenario in SCENARIOS:
            scenario_df = species_df[species_df["scenario"] == scenario].sort_values("time_h")
            final_rows[scenario] = scenario_df.iloc[-1]

        regular_high = float(final_rows["regular"]["high_fraction"])
        dispersed_high = float(final_rows["dispersed_troughs"]["high_fraction"])
        clustered_high = float(final_rows["clustered_troughs"]["high_fraction"])

        regular_total = float(final_rows["regular"]["N_total"])
        dispersed_total = float(final_rows["dispersed_troughs"]["N_total"])
        clustered_total = float(final_rows["clustered_troughs"]["N_total"])

        diff_clustered_minus_dispersed = clustered_high - dispersed_high
        diff_clustered_minus_regular = clustered_high - regular_high
        diff_dispersed_minus_regular = dispersed_high - regular_high

        clustered_gt_dispersed = diff_clustered_minus_dispersed > 0
        clustered_gt_regular = diff_clustered_minus_regular > 0
        dispersed_gt_regular = diff_dispersed_minus_regular > 0

        extinct_any = any(
            x < EXTINCTION_THRESHOLD
            for x in [regular_total, dispersed_total, clustered_total]
        )

        rows.append({
            "parameter_id": parameter_id,
            "species": species,
            "species_short": REPRESENTATIVE_MIC[species]["short_name"],

            "full_dose": params["full_dose"],
            "reduced_dose_fraction": params["reduced_dose_fraction"],
            "reduced_dose": params["reduced_dose"],
            "half_life": params["half_life"],
            "baseline_growth_rate": params["baseline_growth_rate"],
            "max_drug_effect": params["max_drug_effect"],
            "ec50_ratio": params["ec50_ratio"],
            "hill_coefficient": params["hill_coefficient"],

            "final_high_fraction_regular": regular_high,
            "final_high_fraction_dispersed": dispersed_high,
            "final_high_fraction_clustered": clustered_high,

            "final_total_regular": regular_total,
            "final_total_dispersed": dispersed_total,
            "final_total_clustered": clustered_total,

            "diff_clustered_minus_dispersed": diff_clustered_minus_dispersed,
            "diff_clustered_minus_regular": diff_clustered_minus_regular,
            "diff_dispersed_minus_regular": diff_dispersed_minus_regular,

            "clustered_gt_dispersed": clustered_gt_dispersed,
            "clustered_gt_regular": clustered_gt_regular,
            "dispersed_gt_regular": dispersed_gt_regular,
            "extinct_any": extinct_any,
        })

    return pd.DataFrame(rows)


def aggregate_species_summary(per_species_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize sensitivity results across all candidate parameter sets for each species.
    """
    rows = []

    for species, group in per_species_df.groupby("species"):
        diff_cd = group["diff_clustered_minus_dispersed"]
        diff_cr = group["diff_clustered_minus_regular"]
        diff_dr = group["diff_dispersed_minus_regular"]

        rows.append({
            "species": species,
            "species_short": group["species_short"].iloc[0],
            "n_candidate_parameter_sets": len(group),

            "percent_clustered_gt_dispersed": 100.0 * group["clustered_gt_dispersed"].mean(),
            "percent_clustered_gt_regular": 100.0 * group["clustered_gt_regular"].mean(),
            "percent_dispersed_gt_regular": 100.0 * group["dispersed_gt_regular"].mean(),

            "mean_diff_clustered_minus_dispersed": diff_cd.mean(),
            "median_diff_clustered_minus_dispersed": diff_cd.median(),
            "q1_diff_clustered_minus_dispersed": diff_cd.quantile(0.25),
            "q3_diff_clustered_minus_dispersed": diff_cd.quantile(0.75),

            "mean_diff_clustered_minus_regular": diff_cr.mean(),
            "median_diff_clustered_minus_regular": diff_cr.median(),

            "mean_diff_dispersed_minus_regular": diff_dr.mean(),
            "median_diff_dispersed_minus_regular": diff_dr.median(),

            "median_final_high_fraction_regular": group["final_high_fraction_regular"].median(),
            "median_final_high_fraction_dispersed": group["final_high_fraction_dispersed"].median(),
            "median_final_high_fraction_clustered": group["final_high_fraction_clustered"].median(),
        })

    return pd.DataFrame(rows)


def overall_summary(per_species_df: pd.DataFrame) -> pd.DataFrame:
    """
    One overall summary row for the whole deterministic sensitivity analysis.
    """
    rows = [{
        "total_candidate_parameter_sets": per_species_df["parameter_id"].nunique(),
        "total_species_evaluations": len(per_species_df),
        "overall_percent_clustered_gt_dispersed": 100.0 * per_species_df["clustered_gt_dispersed"].mean(),
        "overall_percent_clustered_gt_regular": 100.0 * per_species_df["clustered_gt_regular"].mean(),
        "overall_percent_dispersed_gt_regular": 100.0 * per_species_df["dispersed_gt_regular"].mean(),
        "overall_mean_diff_clustered_minus_dispersed": per_species_df["diff_clustered_minus_dispersed"].mean(),
        "overall_median_diff_clustered_minus_dispersed": per_species_df["diff_clustered_minus_dispersed"].median(),
    }]
    return pd.DataFrame(rows)


# ============================================================
# 8. Figures
# ============================================================

def make_figures(species_summary_df: pd.DataFrame) -> None:
    # Figure 1: % candidate sets with clustered > dispersed
    plot_df = species_summary_df.sort_values("species_short").copy()

    plt.figure(figsize=(8, 5))
    plt.bar(
        plot_df["species_short"],
        plot_df["percent_clustered_gt_dispersed"],
    )
    plt.xlabel("Pathogen")
    plt.ylabel("% candidate parameter sets with clustered > dispersed")
    plt.title("Deterministic sensitivity analysis: clustered vs dispersed")
    plt.ylim(0, 100)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "figure_sensitivity_percent_clustered_gt_dispersed.png",
        dpi=300,
    )
    plt.close()

    # Figure 2: median clustered - dispersed difference
    plt.figure(figsize=(8, 5))
    plt.bar(
        plot_df["species_short"],
        plot_df["median_diff_clustered_minus_dispersed"],
    )
    plt.xlabel("Pathogen")
    plt.ylabel("Median(final high-MIC fraction: clustered - dispersed)")
    plt.title("Median clustered-minus-dispersed difference across candidate sets")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "figure_sensitivity_median_clustered_minus_dispersed.png",
        dpi=300,
    )
    plt.close()


# ============================================================
# 9. Main
# ============================================================

def main() -> None:
    candidate_df = load_candidate_parameter_sets()

    all_per_species = []

    total_candidates = len(candidate_df)
    print("Running deterministic sensitivity analysis...")

    for i, row in candidate_df.iterrows():
        parameter_id = i + 1

        if parameter_id == 1 or parameter_id % 25 == 0:
            print(f"  Processing candidate set {parameter_id}/{total_candidates}")

        params = {
            "full_dose": float(row["full_dose"]),
            "reduced_dose_fraction": float(row["reduced_dose_fraction"]),
            "reduced_dose": float(row["reduced_dose"]),
            "half_life": float(row["half_life"]),
            "baseline_growth_rate": float(row["baseline_growth_rate"]),
            "max_drug_effect": float(row["max_drug_effect"]),
            "ec50_ratio": float(row["ec50_ratio"]),
            "hill_coefficient": float(row["hill_coefficient"]),
        }

        trajectory_df = generate_exposure_trajectories(
            full_dose=params["full_dose"],
            reduced_dose_fraction=params["reduced_dose_fraction"],
            half_life=params["half_life"],
        )

        timeseries_df = simulate_all_conditions(trajectory_df, params)
        per_species_df = summarize_candidate_by_species(
            timeseries_df=timeseries_df,
            parameter_id=parameter_id,
            params=params,
        )
        all_per_species.append(per_species_df)

    per_species_results_df = pd.concat(all_per_species, ignore_index=True)
    species_summary_df = aggregate_species_summary(per_species_results_df)
    overall_summary_df = overall_summary(per_species_results_df)

    # Save outputs
    per_species_path = OUTPUT_DIR / "deterministic_sensitivity_per_parameter_species.csv"
    species_summary_path = OUTPUT_DIR / "deterministic_sensitivity_species_summary.csv"
    overall_summary_path = OUTPUT_DIR / "deterministic_sensitivity_overall_summary.csv"

    per_species_results_df.to_csv(per_species_path, index=False)
    species_summary_df.to_csv(species_summary_path, index=False)
    overall_summary_df.to_csv(overall_summary_path, index=False)

    make_figures(species_summary_df)

    print("\nDeterministic sensitivity analysis completed.")
    print(f"Per-parameter/species results saved to: {per_species_path}")
    print(f"Species summary saved to: {species_summary_path}")
    print(f"Overall summary saved to: {overall_summary_path}")
    print(f"Figures saved to: {OUTPUT_DIR}")
    print("\nSpecies summary preview:")
    print(species_summary_df)


if __name__ == "__main__":
    main()
