#!/usr/bin/env python3
"""
06_stochastic_simulation.py

2.10. Stochastic Simulation of High-MIC Phenotype Establishment and Dominance

Purpose
-------
This script runs stochastic birth-death simulations using the baseline parameter
set selected from deterministic parameter screening.

This is NOT machine-learning training.
It is repeated stochastic simulation.

Required input file in the same folder
--------------------------------------
    best_parameter_set.csv

Optional input file
-------------------
    parameter_screening_candidate_results.csv

The optional file is not used by default. It is only used if:
    RUN_PARAMETER_SENSITIVITY = True

Main outputs
------------
All outputs are saved to:

    stochastic_simulation_output/

Files:
    stochastic_baseline_replicate_outcomes.csv
    stochastic_baseline_summary.csv
    stochastic_baseline_time_quantiles.csv
    figure_stochastic_establishment_probability.png
    figure_stochastic_dominance_probability.png
    figure_stochastic_extinction_probability.png
    figure_stochastic_final_high_fraction.png
    figure_stochastic_<species>_median_high_fraction.png

Optional sensitivity outputs, if RUN_PARAMETER_SENSITIVITY = True:
    stochastic_parameter_sensitivity_outcomes.csv
    stochastic_parameter_sensitivity_summary.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. User-adjustable settings
# ============================================================

# Main baseline stochastic simulation
N_REPLICATES_BASELINE = 1000

# Optional parameter-sensitivity stochastic simulation
# Keep this False for the first run. Set True later if needed.
RUN_PARAMETER_SENSITIVITY = False
N_PARAMETER_SETS_FOR_SENSITIVITY = 30
N_REPLICATES_PER_PARAMETER_SET = 200

# Random seed for reproducibility
RANDOM_SEED = 42


# ============================================================
# 2. File paths
# ============================================================

DATA_DIR = Path(".")
BEST_PARAMETER_FILE = DATA_DIR / "best_parameter_set.csv"
CANDIDATE_PARAMETER_FILE = DATA_DIR / "parameter_screening_candidate_results.csv"

OUTPUT_DIR = DATA_DIR / "stochastic_simulation_output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 3. Representative MIC phenotypes from Table 3
# ============================================================

REPRESENTATIVE_MIC = {
    "Escherichia coli": {
        "short_name": "E. coli",
        "file_name": "e_coli",
        "low_mic": 0.016,
        "high_mic": 0.125,
    },
    "Klebsiella pneumoniae": {
        "short_name": "K. pneumoniae",
        "file_name": "k_pneumoniae",
        "low_mic": 0.03,
        "high_mic": 0.25,
    },
    "Pseudomonas aeruginosa": {
        "short_name": "P. aeruginosa",
        "file_name": "p_aeruginosa",
        "low_mic": 0.125,
        "high_mic": 1.0,
    },
}


# ============================================================
# 4. Fixed simulation settings
# ============================================================

TOTAL_TIME = 72.0
DT = 0.1
DOSE_TIMES = [0, 12, 24, 36, 48, 60]

SCENARIOS = ["regular", "dispersed_troughs", "clustered_troughs"]

INITIAL_TOTAL_POPULATION = 1_000_000
INITIAL_HIGH_FRACTION = 1e-4
CARRYING_CAPACITY = 1e9
HIGH_MIC_FITNESS_COST = 0.05

ESTABLISHMENT_THRESHOLD = 0.01
DOMINANCE_THRESHOLD = 0.50
EXTINCTION_THRESHOLD = 1_000


# ============================================================
# 5. Input loading
# ============================================================

def load_best_parameter_set() -> dict:
    """
    Load the selected baseline parameter set.
    """
    if not BEST_PARAMETER_FILE.exists():
        raise FileNotFoundError(
            "best_parameter_set.csv was not found.\n"
            "Place 'best_parameter_set.csv' in the same folder as this script."
        )

    df = pd.read_csv(BEST_PARAMETER_FILE)

    if len(df) == 0:
        raise ValueError("best_parameter_set.csv is empty.")

    row = df.iloc[0]

    required_columns = [
        "full_dose",
        "reduced_dose_fraction",
        "half_life",
        "baseline_growth_rate",
        "max_drug_effect",
        "ec50_ratio",
        "hill_coefficient",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"best_parameter_set.csv is missing required columns: {missing}"
        )

    full_dose = float(row["full_dose"])
    reduced_dose_fraction = float(row["reduced_dose_fraction"])

    params = {
        "full_dose": full_dose,
        "reduced_dose_fraction": reduced_dose_fraction,
        "reduced_dose": full_dose * reduced_dose_fraction,
        "half_life": float(row["half_life"]),
        "baseline_growth_rate": float(row["baseline_growth_rate"]),
        "max_drug_effect": float(row["max_drug_effect"]),
        "ec50_ratio": float(row["ec50_ratio"]),
        "hill_coefficient": float(row["hill_coefficient"]),
    }

    return params


def load_candidate_parameter_sets() -> pd.DataFrame:
    """
    Load candidate parameter sets for optional stochastic sensitivity analysis.
    """
    if not CANDIDATE_PARAMETER_FILE.exists():
        raise FileNotFoundError(
            "parameter_screening_candidate_results.csv was not found.\n"
            "Set RUN_PARAMETER_SENSITIVITY = False or place the candidate file "
            "in the same folder as this script."
        )

    df = pd.read_csv(CANDIDATE_PARAMETER_FILE)

    if len(df) == 0:
        raise ValueError("parameter_screening_candidate_results.csv is empty.")

    # Use top-scoring parameter sets if score is available.
    if "score" in df.columns:
        df = df.sort_values("score", ascending=False).head(
            N_PARAMETER_SETS_FOR_SENSITIVITY
        )
    else:
        df = df.head(N_PARAMETER_SETS_FOR_SENSITIVITY)

    return df.reset_index(drop=True)


# ============================================================
# 6. Exposure trajectory generation
# ============================================================

def generate_time_points() -> np.ndarray:
    return np.round(np.arange(0, TOTAL_TIME + DT, DT), 6)


def build_dose_patterns(full_dose: float, reduced_dose: float) -> dict:
    """
    Generate the same exposure scenario structure used in deterministic analysis.
    """
    return {
        "regular": [
            full_dose,
            full_dose,
            full_dose,
            full_dose,
            full_dose,
            full_dose,
        ],
        "dispersed_troughs": [
            full_dose,
            reduced_dose,
            full_dose,
            full_dose,
            reduced_dose,
            full_dose,
        ],
        "clustered_troughs": [
            full_dose,
            full_dose,
            reduced_dose,
            reduced_dose,
            full_dose,
            full_dose,
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


def generate_exposure_trajectories(params: dict) -> pd.DataFrame:
    full_dose = params["full_dose"]
    reduced_dose = params["reduced_dose"]
    half_life = params["half_life"]

    dose_patterns = build_dose_patterns(full_dose, reduced_dose)

    trajectory_df = pd.DataFrame({
        "time_h": generate_time_points()
    })

    for scenario, pattern in dose_patterns.items():
        trajectory_df[scenario] = generate_concentration_trajectory(
            dose_pattern=pattern,
            half_life=half_life,
        )

    return trajectory_df


# ============================================================
# 7. Stochastic model functions
# ============================================================

def drug_effect(
    concentration: float,
    mic: float,
    max_drug_effect: float,
    ec50_ratio: float,
    hill_coefficient: float,
) -> float:
    """
    Emax-type drug effect as a function of exposure ratio C(t)/MIC.
    """
    exposure_ratio = concentration / mic
    numerator = exposure_ratio ** hill_coefficient
    denominator = (ec50_ratio ** hill_coefficient) + numerator

    if denominator <= 0:
        return 0.0

    return max_drug_effect * numerator / denominator


def simulate_stochastic_condition_vectorized(
    trajectory_df: pd.DataFrame,
    species: str,
    scenario: str,
    low_mic: float,
    high_mic: float,
    params: dict,
    n_replicates: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run stochastic tau-leaping birth-death simulations for one species and one scenario.

    The deterministic net growth structure is converted into stochastic events:
        births ~ Poisson(N * growth_rate * dt)
        deaths ~ Poisson(N * drug_effect * dt)

    This approximates random birth and death events while remaining fast enough
    for many repeated simulations.
    """
    time_values = trajectory_df["time_h"].to_numpy()
    concentration_values = trajectory_df[scenario].to_numpy()

    n_time = len(time_values)

    # Integer initial populations
    initial_high = int(round(INITIAL_TOTAL_POPULATION * INITIAL_HIGH_FRACTION))
    initial_low = int(INITIAL_TOTAL_POPULATION - initial_high)

    n_low = np.full(n_replicates, initial_low, dtype=np.int64)
    n_high = np.full(n_replicates, initial_high, dtype=np.int64)

    high_fraction_matrix = np.zeros((n_replicates, n_time), dtype=np.float64)

    established_ever = np.zeros(n_replicates, dtype=bool)
    dominated_ever = np.zeros(n_replicates, dtype=bool)
    extinct_ever = np.zeros(n_replicates, dtype=bool)

    time_to_establishment = np.full(n_replicates, np.nan, dtype=np.float64)
    time_to_dominance = np.full(n_replicates, np.nan, dtype=np.float64)
    time_to_extinction = np.full(n_replicates, np.nan, dtype=np.float64)

    for idx, time_h in enumerate(time_values):
        concentration = float(concentration_values[idx])

        n_total = n_low + n_high

        high_fraction = np.divide(
            n_high,
            n_total,
            out=np.zeros_like(n_high, dtype=np.float64),
            where=n_total > 0,
        )

        high_fraction_matrix[:, idx] = high_fraction

        newly_established = (
            (high_fraction >= ESTABLISHMENT_THRESHOLD)
            & (~established_ever)
            & (n_total >= EXTINCTION_THRESHOLD)
        )
        time_to_establishment[newly_established] = time_h
        established_ever[newly_established] = True

        newly_dominated = (
            (high_fraction >= DOMINANCE_THRESHOLD)
            & (~dominated_ever)
            & (n_total >= EXTINCTION_THRESHOLD)
        )
        time_to_dominance[newly_dominated] = time_h
        dominated_ever[newly_dominated] = True

        newly_extinct = (n_total < EXTINCTION_THRESHOLD) & (~extinct_ever)
        time_to_extinction[newly_extinct] = time_h
        extinct_ever[newly_extinct] = True

        if idx == n_time - 1:
            break

        # Current total after checks
        n_total = n_low + n_high

        logistic_factor = np.maximum(
            0.0,
            1.0 - (n_total.astype(np.float64) / CARRYING_CAPACITY),
        )

        growth_low = params["baseline_growth_rate"] * logistic_factor
        growth_high = (
            params["baseline_growth_rate"]
            * (1.0 - HIGH_MIC_FITNESS_COST)
            * logistic_factor
        )

        effect_low = drug_effect(
            concentration=concentration,
            mic=low_mic,
            max_drug_effect=params["max_drug_effect"],
            ec50_ratio=params["ec50_ratio"],
            hill_coefficient=params["hill_coefficient"],
        )
        effect_high = drug_effect(
            concentration=concentration,
            mic=high_mic,
            max_drug_effect=params["max_drug_effect"],
            ec50_ratio=params["ec50_ratio"],
            hill_coefficient=params["hill_coefficient"],
        )

        # Tau-leaping birth and death events
        birth_low_lambda = np.maximum(n_low * growth_low * DT, 0.0)
        birth_high_lambda = np.maximum(n_high * growth_high * DT, 0.0)

        death_low_lambda = np.maximum(n_low * effect_low * DT, 0.0)
        death_high_lambda = np.maximum(n_high * effect_high * DT, 0.0)

        births_low = rng.poisson(birth_low_lambda)
        births_high = rng.poisson(birth_high_lambda)

        deaths_low = rng.poisson(death_low_lambda)
        deaths_high = rng.poisson(death_high_lambda)

        # Deaths cannot exceed current population
        deaths_low = np.minimum(deaths_low, n_low)
        deaths_high = np.minimum(deaths_high, n_high)

        n_low = n_low + births_low - deaths_low
        n_high = n_high + births_high - deaths_high

        # Prevent negative values
        n_low = np.maximum(n_low, 0)
        n_high = np.maximum(n_high, 0)

    final_total = n_low + n_high
    final_high_fraction = np.divide(
        n_high,
        final_total,
        out=np.zeros_like(n_high, dtype=np.float64),
        where=final_total > 0,
    )

    max_high_fraction = np.max(high_fraction_matrix, axis=1)

    replicate_outcomes = pd.DataFrame({
        "species": species,
        "scenario": scenario,
        "replicate": np.arange(1, n_replicates + 1),
        "final_N_low": n_low,
        "final_N_high": n_high,
        "final_N_total": final_total,
        "final_high_fraction": final_high_fraction,
        "max_high_fraction": max_high_fraction,
        "established_ever": established_ever,
        "dominated_ever": dominated_ever,
        "extinct_ever": extinct_ever,
        "time_to_establishment_h": time_to_establishment,
        "time_to_dominance_h": time_to_dominance,
        "time_to_extinction_h": time_to_extinction,
    })

    quantiles = np.quantile(
        high_fraction_matrix,
        q=[0.10, 0.25, 0.50, 0.75, 0.90],
        axis=0,
    )

    time_quantiles = pd.DataFrame({
        "species": species,
        "scenario": scenario,
        "time_h": time_values,
        "high_fraction_q10": quantiles[0],
        "high_fraction_q25": quantiles[1],
        "high_fraction_median": quantiles[2],
        "high_fraction_q75": quantiles[3],
        "high_fraction_q90": quantiles[4],
    })

    return replicate_outcomes, time_quantiles


# ============================================================
# 8. Summary functions
# ============================================================

def summarize_replicate_outcomes(outcomes_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (species, scenario), group in outcomes_df.groupby(["species", "scenario"]):
        valid_est_times = group["time_to_establishment_h"].dropna()
        valid_dom_times = group["time_to_dominance_h"].dropna()

        rows.append({
            "species": species,
            "scenario": scenario,
            "n_replicates": len(group),

            "P_establishment": group["established_ever"].mean(),
            "P_dominance": group["dominated_ever"].mean(),
            "P_extinction": group["extinct_ever"].mean(),

            "median_final_high_fraction": group["final_high_fraction"].median(),
            "q25_final_high_fraction": group["final_high_fraction"].quantile(0.25),
            "q75_final_high_fraction": group["final_high_fraction"].quantile(0.75),

            "median_final_total_population": group["final_N_total"].median(),
            "q25_final_total_population": group["final_N_total"].quantile(0.25),
            "q75_final_total_population": group["final_N_total"].quantile(0.75),

            "median_time_to_establishment_h": (
                valid_est_times.median() if len(valid_est_times) > 0 else np.nan
            ),
            "median_time_to_dominance_h": (
                valid_dom_times.median() if len(valid_dom_times) > 0 else np.nan
            ),
        })

    summary_df = pd.DataFrame(rows)

    scenario_order = {
        "regular": 0,
        "dispersed_troughs": 1,
        "clustered_troughs": 2,
    }
    species_order = {
        "Escherichia coli": 0,
        "Klebsiella pneumoniae": 1,
        "Pseudomonas aeruginosa": 2,
    }

    summary_df["species_order"] = summary_df["species"].map(species_order)
    summary_df["scenario_order"] = summary_df["scenario"].map(scenario_order)
    summary_df = summary_df.sort_values(
        ["species_order", "scenario_order"]
    ).drop(columns=["species_order", "scenario_order"])

    return summary_df


def compare_clustered_vs_dispersed(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for species, group in summary_df.groupby("species"):
        dispersed = group[group["scenario"] == "dispersed_troughs"].iloc[0]
        clustered = group[group["scenario"] == "clustered_troughs"].iloc[0]
        regular = group[group["scenario"] == "regular"].iloc[0]

        rows.append({
            "species": species,

            "delta_P_establishment_clustered_minus_dispersed": (
                clustered["P_establishment"] - dispersed["P_establishment"]
            ),
            "delta_P_dominance_clustered_minus_dispersed": (
                clustered["P_dominance"] - dispersed["P_dominance"]
            ),
            "delta_median_final_high_fraction_clustered_minus_dispersed": (
                clustered["median_final_high_fraction"]
                - dispersed["median_final_high_fraction"]
            ),

            "delta_P_establishment_dispersed_minus_regular": (
                dispersed["P_establishment"] - regular["P_establishment"]
            ),
            "delta_P_dominance_dispersed_minus_regular": (
                dispersed["P_dominance"] - regular["P_dominance"]
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# 9. Plotting functions
# ============================================================

def make_probability_bar_figure(
    summary_df: pd.DataFrame,
    value_column: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    species_order = [
        "Escherichia coli",
        "Klebsiella pneumoniae",
        "Pseudomonas aeruginosa",
    ]
    species_labels = ["E. coli", "K. pneumoniae", "P. aeruginosa"]
    scenarios = ["regular", "dispersed_troughs", "clustered_troughs"]
    scenario_labels = ["Regular", "Dispersed", "Clustered"]

    x = np.arange(len(species_order))
    width = 0.25

    plt.figure(figsize=(10, 6))

    for i, scenario in enumerate(scenarios):
        values = []
        for species in species_order:
            value = summary_df.loc[
                (summary_df["species"] == species)
                & (summary_df["scenario"] == scenario),
                value_column,
            ].values[0]
            values.append(value)

        plt.bar(
            x + (i - 1) * width,
            values,
            width,
            label=scenario_labels[i],
        )

    plt.xticks(x, species_labels)
    plt.xlabel("Pathogen")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


def make_final_fraction_bar_figure(summary_df: pd.DataFrame) -> None:
    make_probability_bar_figure(
        summary_df=summary_df,
        value_column="median_final_high_fraction",
        ylabel="Median final high-MIC phenotype fraction",
        title="Stochastic simulation: median final high-MIC phenotype fraction",
        filename="figure_stochastic_final_high_fraction.png",
    )


def make_median_trajectory_figures(time_quantiles_df: pd.DataFrame) -> None:
    scenario_label_map = {
        "regular": "Regular exposure",
        "dispersed_troughs": "Dispersed troughs",
        "clustered_troughs": "Clustered troughs",
    }

    for species, info in REPRESENTATIVE_MIC.items():
        species_df = time_quantiles_df[time_quantiles_df["species"] == species]

        plt.figure(figsize=(10, 6))

        for scenario in SCENARIOS:
            plot_df = species_df[species_df["scenario"] == scenario]
            plt.plot(
                plot_df["time_h"],
                plot_df["high_fraction_median"],
                label=scenario_label_map[scenario],
            )

        plt.xlabel("Time (h)")
        plt.ylabel("Median high-MIC phenotype fraction")
        plt.title(
            f"Stochastic median high-MIC fraction dynamics in {info['short_name']}"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f"figure_stochastic_{info['file_name']}_median_high_fraction.png"
        plt.savefig(OUTPUT_DIR / filename, dpi=300)
        plt.close()


# ============================================================
# 10. Baseline stochastic simulation
# ============================================================

def run_baseline_stochastic_simulation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    params = load_best_parameter_set()

    print("\nLoaded baseline parameter set:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    trajectory_df = generate_exposure_trajectories(params)

    # Save the baseline stochastic exposure trajectories for documentation
    trajectory_df.to_csv(
        OUTPUT_DIR / "stochastic_baseline_exposure_trajectories.csv",
        index=False,
    )

    rng = np.random.default_rng(RANDOM_SEED)

    all_outcomes = []
    all_time_quantiles = []

    total_conditions = len(REPRESENTATIVE_MIC) * len(SCENARIOS)
    condition_counter = 0

    print("\nRunning baseline stochastic simulations...")

    for species, mic_info in REPRESENTATIVE_MIC.items():
        for scenario in SCENARIOS:
            condition_counter += 1
            print(
                f"  Condition {condition_counter}/{total_conditions}: "
                f"{mic_info['short_name']} - {scenario}"
            )

            outcomes, quantiles = simulate_stochastic_condition_vectorized(
                trajectory_df=trajectory_df,
                species=species,
                scenario=scenario,
                low_mic=mic_info["low_mic"],
                high_mic=mic_info["high_mic"],
                params=params,
                n_replicates=N_REPLICATES_BASELINE,
                rng=rng,
            )

            all_outcomes.append(outcomes)
            all_time_quantiles.append(quantiles)

    outcomes_df = pd.concat(all_outcomes, ignore_index=True)
    time_quantiles_df = pd.concat(all_time_quantiles, ignore_index=True)

    summary_df = summarize_replicate_outcomes(outcomes_df)
    comparison_df = compare_clustered_vs_dispersed(summary_df)

    # Save outputs
    outcomes_df.to_csv(
        OUTPUT_DIR / "stochastic_baseline_replicate_outcomes.csv",
        index=False,
    )
    summary_df.to_csv(
        OUTPUT_DIR / "stochastic_baseline_summary.csv",
        index=False,
    )
    comparison_df.to_csv(
        OUTPUT_DIR / "stochastic_baseline_clustered_vs_dispersed.csv",
        index=False,
    )
    time_quantiles_df.to_csv(
        OUTPUT_DIR / "stochastic_baseline_time_quantiles.csv",
        index=False,
    )

    # Figures
    make_probability_bar_figure(
        summary_df=summary_df,
        value_column="P_establishment",
        ylabel="Probability of establishment",
        title="Stochastic simulation: high-MIC establishment probability",
        filename="figure_stochastic_establishment_probability.png",
    )

    make_probability_bar_figure(
        summary_df=summary_df,
        value_column="P_dominance",
        ylabel="Probability of dominance",
        title="Stochastic simulation: high-MIC dominance probability",
        filename="figure_stochastic_dominance_probability.png",
    )

    make_probability_bar_figure(
        summary_df=summary_df,
        value_column="P_extinction",
        ylabel="Probability of extinction",
        title="Stochastic simulation: population extinction probability",
        filename="figure_stochastic_extinction_probability.png",
    )

    make_final_fraction_bar_figure(summary_df)
    make_median_trajectory_figures(time_quantiles_df)

    print("\nBaseline stochastic simulation completed.")
    print(f"Outputs saved to: {OUTPUT_DIR}")

    print("\nSummary preview:")
    print(summary_df)

    print("\nClustered vs dispersed comparison:")
    print(comparison_df)

    return outcomes_df, summary_df, time_quantiles_df


# ============================================================
# 11. Optional stochastic parameter-sensitivity simulation
# ============================================================

def run_parameter_sensitivity_stochastic_simulation() -> None:
    candidate_df = load_candidate_parameter_sets()
    rng = np.random.default_rng(RANDOM_SEED + 1000)

    all_rows = []

    print("\nRunning optional stochastic parameter-sensitivity simulation...")
    print(
        f"Parameter sets: {len(candidate_df)}, "
        f"replicates per condition: {N_REPLICATES_PER_PARAMETER_SET}"
    )

    for param_idx, row in candidate_df.iterrows():
        parameter_id = param_idx + 1

        if parameter_id == 1 or parameter_id % 5 == 0:
            print(f"  Parameter set {parameter_id}/{len(candidate_df)}")

        full_dose = float(row["full_dose"])
        reduced_dose_fraction = float(row["reduced_dose_fraction"])

        params = {
            "full_dose": full_dose,
            "reduced_dose_fraction": reduced_dose_fraction,
            "reduced_dose": full_dose * reduced_dose_fraction,
            "half_life": float(row["half_life"]),
            "baseline_growth_rate": float(row["baseline_growth_rate"]),
            "max_drug_effect": float(row["max_drug_effect"]),
            "ec50_ratio": float(row["ec50_ratio"]),
            "hill_coefficient": float(row["hill_coefficient"]),
        }

        trajectory_df = generate_exposure_trajectories(params)

        condition_summaries = []

        for species, mic_info in REPRESENTATIVE_MIC.items():
            for scenario in SCENARIOS:
                outcomes, _ = simulate_stochastic_condition_vectorized(
                    trajectory_df=trajectory_df,
                    species=species,
                    scenario=scenario,
                    low_mic=mic_info["low_mic"],
                    high_mic=mic_info["high_mic"],
                    params=params,
                    n_replicates=N_REPLICATES_PER_PARAMETER_SET,
                    rng=rng,
                )

                summary = summarize_replicate_outcomes(outcomes)
                summary["parameter_id"] = parameter_id

                for key, value in params.items():
                    summary[key] = value

                condition_summaries.append(summary)

        parameter_summary = pd.concat(condition_summaries, ignore_index=True)
        all_rows.append(parameter_summary)

    sensitivity_df = pd.concat(all_rows, ignore_index=True)

    sensitivity_df.to_csv(
        OUTPUT_DIR / "stochastic_parameter_sensitivity_outcomes.csv",
        index=False,
    )

    # Summarize clustered vs dispersed across parameter sets
    comparison_rows = []

    for parameter_id, parameter_group in sensitivity_df.groupby("parameter_id"):
        comparison_df = compare_clustered_vs_dispersed(parameter_group)
        comparison_df["parameter_id"] = parameter_id
        comparison_rows.append(comparison_df)

    comparison_all = pd.concat(comparison_rows, ignore_index=True)

    species_summary_rows = []
    for species, group in comparison_all.groupby("species"):
        species_summary_rows.append({
            "species": species,
            "n_parameter_sets": group["parameter_id"].nunique(),

            "percent_clustered_higher_P_establishment": (
                100.0
                * (
                    group["delta_P_establishment_clustered_minus_dispersed"] > 0
                ).mean()
            ),
            "percent_clustered_higher_P_dominance": (
                100.0
                * (
                    group["delta_P_dominance_clustered_minus_dispersed"] > 0
                ).mean()
            ),
            "median_delta_P_establishment_clustered_minus_dispersed": (
                group["delta_P_establishment_clustered_minus_dispersed"].median()
            ),
            "median_delta_P_dominance_clustered_minus_dispersed": (
                group["delta_P_dominance_clustered_minus_dispersed"].median()
            ),
            "median_delta_final_high_fraction_clustered_minus_dispersed": (
                group[
                    "delta_median_final_high_fraction_clustered_minus_dispersed"
                ].median()
            ),
        })

    species_sensitivity_summary = pd.DataFrame(species_summary_rows)

    comparison_all.to_csv(
        OUTPUT_DIR / "stochastic_parameter_sensitivity_clustered_vs_dispersed.csv",
        index=False,
    )
    species_sensitivity_summary.to_csv(
        OUTPUT_DIR / "stochastic_parameter_sensitivity_summary.csv",
        index=False,
    )

    print("\nOptional stochastic parameter-sensitivity simulation completed.")
    print(
        "Sensitivity outputs saved to:\n"
        "  stochastic_parameter_sensitivity_outcomes.csv\n"
        "  stochastic_parameter_sensitivity_clustered_vs_dispersed.csv\n"
        "  stochastic_parameter_sensitivity_summary.csv"
    )


# ============================================================
# 12. Main
# ============================================================

def main() -> None:
    run_baseline_stochastic_simulation()

    if RUN_PARAMETER_SENSITIVITY:
        run_parameter_sensitivity_stochastic_simulation()

    print("\nDone.")
    print("\nImportant output files to inspect first:")
    print("  stochastic_baseline_summary.csv")
    print("  stochastic_baseline_clustered_vs_dispersed.csv")
    print("  figure_stochastic_establishment_probability.png")
    print("  figure_stochastic_dominance_probability.png")
    print("  figure_stochastic_final_high_fraction.png")


if __name__ == "__main__":
    main()
