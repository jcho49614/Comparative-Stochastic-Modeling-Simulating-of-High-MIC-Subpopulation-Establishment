from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("parameter_screening_output")
OUTPUT_DIR.mkdir(exist_ok=True)

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

TOTAL_TIME = 72.0               # h
DT = 0.1                        # h
DOSE_INTERVAL = 12.0            # h
DOSE_TIMES = [0, 12, 24, 36, 48, 60]

SCENARIOS = ["regular", "dispersed_troughs", "clustered_troughs"]

INITIAL_TOTAL_POPULATION = 1e6  # cells
INITIAL_HIGH_FRACTION = 1e-4    # model parameter, not EUCAST fraction
CARRYING_CAPACITY = 1e9         # cells

HIGH_MIC_FITNESS_COST = 0.05    # 5% reduction in high-MIC baseline growth

#lower bound, lower than this considered extinct.
NUMERICAL_FLOOR = 1e-300

EXTINCTION_THRESHOLD = 1e3       # cells; final population below this is considered extinct
SATURATION_THRESHOLD = 0.98      # final population above 98% of K is considered saturated

MIN_FINAL_HIGH_FRACTION_FOR_SIGNAL = 1e-3
MAX_FINAL_HIGH_FRACTION_FOR_SIGNAL = 0.95

MIN_HIGH_FRACTION_RANGE = 0.01   # at least some visible scenario/species variation
MIN_CLUSTERED_MINUS_DISPERSED = 0.005

#maybe expand if time?
PARAMETER_GRID = {
    "full_dose": [0.5, 1.0, 2.0, 4.0],
    "reduced_dose_fraction": [0.25, 0.5],
    "half_life": [4.0, 6.0],
    "baseline_growth_rate": [0.5, 0.8],
    "max_drug_effect": [0.4, 0.6, 0.8, 1.0, 1.2],
    "ec50_ratio": [2.0, 4.0, 8.0, 16.0],
    "hill_coefficient": [1.0, 2.0],
}

def generate_time_points() -> np.ndarray:
    return np.round(np.arange(0, TOTAL_TIME + DT, DT), 6)


def build_dose_patterns(full_dose: float, reduced_dose: float) -> dict:
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


def generate_concentration_trajectory(
    dose_pattern: list,
    half_life: float,
) -> np.ndarray:
    time_points = generate_time_points()
    dose_dict = dict(zip(DOSE_TIMES, dose_pattern))
    decay_constant = np.log(2) / half_life

    concentrations = []
    concentration = 0.0

    for t in time_points:
        #direct matching here
        if float(t) in dose_dict:
            concentration += dose_dict[float(t)]

        concentrations.append(concentration)

        #decay before next step
        concentration = concentration * np.exp(-decay_constant * DT)

    return np.array(concentrations)


def generate_exposure_trajectories(
    full_dose: float,
    reduced_dose_fraction: float,
    half_life: float,
) -> pd.DataFrame:
    reduced_dose = full_dose * reduced_dose_fraction
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

#deterministic model
def drug_effect(
    concentration: float,
    mic: float,
    max_drug_effect: float,
    ec50_ratio: float,
    hill_coefficient: float,
) -> float:
    #Emax-type drug effect as a function of exposure ratio C(t)/MIC.
    if mic <= 0:
        raise ValueError("MIC must be greater than zero.")

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
    
    #low, high MIC values for one species.
    time_values = trajectory_df["time_h"].to_numpy()
    concentration_values = trajectory_df[scenario].to_numpy()

    if len(time_values) < 2:
        raise ValueError("Trajectory must contain at least two time points.")

    dt = float(np.median(np.diff(time_values)))

    n_high = INITIAL_TOTAL_POPULATION * INITIAL_HIGH_FRACTION
    n_low = INITIAL_TOTAL_POPULATION - n_high

    rows = []

    for idx, time_h in enumerate(time_values):
        concentration = float(concentration_values[idx])
        n_total = n_low + n_high

        high_fraction = n_high / n_total if n_total > 0 else np.nan

        effect_low = drug_effect(
            concentration=concentration,
            mic=low_mic,
            max_drug_effect=max_drug_effect,
            ec50_ratio=ec50_ratio,
            hill_coefficient=hill_coefficient,
        )
        effect_high = drug_effect(
            concentration=concentration,
            mic=high_mic,
            max_drug_effect=max_drug_effect,
            ec50_ratio=ec50_ratio,
            hill_coefficient=hill_coefficient,
        )

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
            "drug_effect_low": effect_low,
            "drug_effect_high": effect_high,
            "exposure_ratio_low": concentration / low_mic,
            "exposure_ratio_high": concentration / high_mic,
        })

        if idx == len(time_values) - 1:
            break

        # limitation
        logistic_factor = max(0.0, 1.0 - (n_total / CARRYING_CAPACITY))

        growth_low = baseline_growth_rate * logistic_factor
        growth_high = (
            baseline_growth_rate
            * (1.0 - HIGH_MIC_FITNESS_COST)
            * logistic_factor
        )

        net_growth_low = growth_low - effect_low
        net_growth_high = growth_high - effect_high

        n_low = n_low * np.exp(net_growth_low * dt)
        n_high = n_high * np.exp(net_growth_high * dt)

        #a floor
        n_low = max(n_low, NUMERICAL_FLOOR)
        n_high = max(n_high, NUMERICAL_FLOOR)

    return pd.DataFrame(rows)


def simulate_all_conditions(
    trajectory_df: pd.DataFrame,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
   #run all 3 species
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

    timeseries_df = pd.concat(all_ts, ignore_index=True)
    summary_df = summarize_timeseries(timeseries_df)

    return timeseries_df, summary_df


def summarize_timeseries(timeseries_df: pd.DataFrame) -> pd.DataFrame:
    #summarize deterministic outputs
    rows = []

    for (species, scenario), group in timeseries_df.groupby(["species", "scenario"]):
        group = group.sort_values("time_h")
        initial = group.iloc[0]
        final = group.iloc[-1]

        initial_high_fraction = float(initial["high_fraction"])
        final_high_fraction = float(final["high_fraction"])
        final_total = float(final["N_total"])

        low_mic = float(final["low_mic"])
        high_mic = float(final["high_mic"])

        selective_window_mask = (
            (group["concentration_mg_l"] >= low_mic)
            & (group["concentration_mg_l"] < high_mic)
        )

        time_values = group["time_h"].to_numpy()
        dt = float(np.median(np.diff(time_values))) if len(time_values) > 1 else np.nan
        time_in_selective_window = float(selective_window_mask.sum() * dt)

        auc_0_72 = float(
            np.trapz(group["concentration_mg_l"], group["time_h"])
        )

        rows.append({
            "species": species,
            "scenario": scenario,
            "low_mic": low_mic,
            "high_mic": high_mic,
            "initial_total_population": float(initial["N_total"]),
            "final_total_population": final_total,
            "initial_high_fraction": initial_high_fraction,
            "final_high_fraction": final_high_fraction,
            "fold_change_high_fraction": (
                final_high_fraction / initial_high_fraction
                if initial_high_fraction > 0
                else np.nan
            ),
            "max_high_fraction": float(group["high_fraction"].max()),
            "time_max_high_fraction_h": float(
                group.loc[group["high_fraction"].idxmax(), "time_h"]
            ),
            "AUC_0_72": auc_0_72,
            "time_in_selective_window_h": time_in_selective_window,
            "extinct_at_72h": final_total < EXTINCTION_THRESHOLD,
            "saturated_at_72h": final_total > (SATURATION_THRESHOLD * CARRYING_CAPACITY),
        })

    return pd.DataFrame(rows)

def evaluate_parameter_set(summary_df: pd.DataFrame) -> dict:

    #evaluate the parapmeters
    final_totals = summary_df["final_total_population"].to_numpy()
    final_high = summary_df["final_high_fraction"].to_numpy()

    extinction_count = int((final_totals < EXTINCTION_THRESHOLD).sum())
    saturation_count = int(
        (final_totals > SATURATION_THRESHOLD * CARRYING_CAPACITY).sum()
    )

    no_extinction = extinction_count == 0

    usable_signal_mask = (
        (final_high >= MIN_FINAL_HIGH_FRACTION_FOR_SIGNAL)
        & (final_high <= MAX_FINAL_HIGH_FRACTION_FOR_SIGNAL)
        & (final_totals >= EXTINCTION_THRESHOLD)
    )
    usable_signal_count = int(usable_signal_mask.sum())

    high_fraction_range = float(np.nanmax(final_high) - np.nanmin(final_high))

    # clustered/dispersed comparison by species
    clustered_positive_count = 0
    clustered_minus_dispersed_values = []

    for species in REPRESENTATIVE_MIC.keys():
        species_df = summary_df[summary_df["species"] == species]

        dispersed_value = float(
            species_df.loc[
                species_df["scenario"] == "dispersed_troughs",
                "final_high_fraction",
            ].iloc[0]
        )
        clustered_value = float(
            species_df.loc[
                species_df["scenario"] == "clustered_troughs",
                "final_high_fraction",
            ].iloc[0]
        )

        diff = clustered_value - dispersed_value
        clustered_minus_dispersed_values.append(diff)

        if diff >= MIN_CLUSTERED_MINUS_DISPERSED:
            clustered_positive_count += 1

    mean_clustered_minus_dispersed = float(np.mean(clustered_minus_dispersed_values))
    max_clustered_minus_dispersed = float(np.max(clustered_minus_dispersed_values))

    has_visible_variation = high_fraction_range >= MIN_HIGH_FRACTION_RANGE
    has_some_usable_signal = usable_signal_count >= 3

   
    is_candidate = (
        no_extinction
        and has_visible_variation
        and has_some_usable_signal
    )

    # score for usable candidate
    score = 0.0

    if no_extinction:
        score += 1000.0
    else:
        score -= 150.0 * extinction_count

    score += 50.0 * usable_signal_count
    score += 300.0 * min(high_fraction_range / 0.25, 1.0)
    score += 150.0 * (clustered_positive_count / 3.0)
    score += 200.0 * max(0.0, min(mean_clustered_minus_dispersed / 0.10, 1.0))
    score -= 25.0 * saturation_count

    if high_fraction_range < MIN_HIGH_FRACTION_RANGE:
        score -= 200.0

    return {
        "is_candidate": is_candidate,
        "score": score,
        "extinction_count": extinction_count,
        "saturation_count": saturation_count,
        "usable_signal_count": usable_signal_count,
        "high_fraction_range": high_fraction_range,
        "clustered_positive_count": clustered_positive_count,
        "mean_clustered_minus_dispersed": mean_clustered_minus_dispersed,
        "max_clustered_minus_dispersed": max_clustered_minus_dispersed,
    }

def parameter_combinations() -> list:
    """Generate all parameter combinations from PARAMETER_GRID."""
    combos = []

    for full_dose in PARAMETER_GRID["full_dose"]:
        for reduced_dose_fraction in PARAMETER_GRID["reduced_dose_fraction"]:
            for half_life in PARAMETER_GRID["half_life"]:
                for baseline_growth_rate in PARAMETER_GRID["baseline_growth_rate"]:
                    for max_drug_effect in PARAMETER_GRID["max_drug_effect"]:
                        for ec50_ratio in PARAMETER_GRID["ec50_ratio"]:
                            for hill_coefficient in PARAMETER_GRID["hill_coefficient"]:
                                combos.append({
                                    "full_dose": full_dose,
                                    "reduced_dose_fraction": reduced_dose_fraction,
                                    "reduced_dose": full_dose * reduced_dose_fraction,
                                    "half_life": half_life,
                                    "baseline_growth_rate": baseline_growth_rate,
                                    "max_drug_effect": max_drug_effect,
                                    "ec50_ratio": ec50_ratio,
                                    "hill_coefficient": hill_coefficient,
                                })

    return combos


def run_screening() -> pd.DataFrame:
    """
    Run deterministic screening across the full parameter grid.
    """
    combos = parameter_combinations()
    all_results = []

    print(f"Total parameter combinations: {len(combos)}")
    print("Running deterministic parameter screening...")

    for idx, params in enumerate(combos, start=1):
        if idx % 50 == 0 or idx == 1:
            print(f"  Running combination {idx}/{len(combos)}")

        trajectory_df = generate_exposure_trajectories(
            full_dose=params["full_dose"],
            reduced_dose_fraction=params["reduced_dose_fraction"],
            half_life=params["half_life"],
        )

        _, summary_df = simulate_all_conditions(
            trajectory_df=trajectory_df,
            params=params,
        )

        evaluation = evaluate_parameter_set(summary_df)

        result_row = {
            **params,
            **evaluation,
        }

        all_results.append(result_row)

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(
        ["is_candidate", "score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return results_df


#save the best output function

def save_best_parameter_outputs(best_params: dict) -> None:
    """
    re-running the deterministic model for the selected best parameter set and
    save trajectories, population outputs, summary tables, and figures.
    """
    trajectory_df = generate_exposure_trajectories(
        full_dose=best_params["full_dose"],
        reduced_dose_fraction=best_params["reduced_dose_fraction"],
        half_life=best_params["half_life"],
    )

    timeseries_df, summary_df = simulate_all_conditions(
        trajectory_df=trajectory_df,
        params=best_params,
    )

    #output stuff
    trajectory_path = OUTPUT_DIR / "best_exposure_trajectories.csv"
    timeseries_path = OUTPUT_DIR / "best_deterministic_population_timeseries.csv"
    summary_path = OUTPUT_DIR / "best_deterministic_population_summary.csv"
    best_params_path = OUTPUT_DIR / "best_parameter_set.csv"
    best_params_json_path = OUTPUT_DIR / "best_parameter_set.json"

    trajectory_df.to_csv(trajectory_path, index=False)
    timeseries_df.to_csv(timeseries_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pd.DataFrame([best_params]).to_csv(best_params_path, index=False)

    with best_params_json_path.open("w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=4)

    # each species high mic map
    scenario_label_map = {
        "regular": "Regular exposure",
        "dispersed_troughs": "Dispersed troughs",
        "clustered_troughs": "Clustered troughs",
    }

    for species, info in REPRESENTATIVE_MIC.items():
        species_df = timeseries_df[timeseries_df["species"] == species]

        plt.figure(figsize=(10, 6))

        for scenario in SCENARIOS:
            plot_df = species_df[species_df["scenario"] == scenario]
            plt.plot(
                plot_df["time_h"],
                plot_df["high_fraction"],
                label=scenario_label_map[scenario],
            )

        plt.xlabel("Time (h)")
        plt.ylabel("High-MIC phenotype fraction")
        plt.title(
            f"Deterministic high-MIC fraction dynamics in {info['short_name']}"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_path = OUTPUT_DIR / (
            f"figure_best_{info['file_name']}_high_mic_fraction.png"
        )
        plt.savefig(fig_path, dpi=300)
        plt.close()

   #figure with barchart for easy visualization
    plot_summary = summary_df.copy()
    plot_summary["species_short"] = plot_summary["species"].map(
        {s: info["short_name"] for s, info in REPRESENTATIVE_MIC.items()}
    )
    plot_summary["scenario_label"] = plot_summary["scenario"].map(scenario_label_map)

    species_list = [info["short_name"] for info in REPRESENTATIVE_MIC.values()]
    x = np.arange(len(species_list))
    width = 0.25

    plt.figure(figsize=(10, 6))

    for i, scenario in enumerate(SCENARIOS):
        scenario_df = plot_summary[plot_summary["scenario"] == scenario]
        values = []

        for species_short in species_list:
            value = scenario_df.loc[
                scenario_df["species_short"] == species_short,
                "final_high_fraction",
            ].values[0]
            values.append(value)

        plt.bar(
            x + (i - 1) * width,
            values,
            width,
            label=scenario_label_map[scenario],
        )

    plt.xticks(x, species_list)
    plt.xlabel("Pathogen")
    plt.ylabel("Final high-MIC phenotype fraction")
    plt.title("Final high-MIC phenotype fraction at 72 h")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    final_fig_path = OUTPUT_DIR / "figure_best_final_high_mic_fraction.png"
    plt.savefig(final_fig_path, dpi=300)
    plt.close()

    print("\nBest parameter outputs saved:")
    print(f"  {trajectory_path}")
    print(f"  {timeseries_path}")
    print(f"  {summary_path}")
    print(f"  {best_params_path}")
    print(f"  {best_params_json_path}")
    print(f"  Figures saved in: {OUTPUT_DIR}")


#main func
def main() -> None:
    results_df = run_screening()

    all_results_path = OUTPUT_DIR / "parameter_screening_all_results.csv"
    candidate_results_path = OUTPUT_DIR / "parameter_screening_candidate_results.csv"

    results_df.to_csv(all_results_path, index=False)

    candidates_df = results_df[results_df["is_candidate"] == True].copy()
    candidates_df.to_csv(candidate_results_path, index=False)

    print("\nScreening completed.")
    print(f"All results saved to: {all_results_path}")
    print(f"Candidate results saved to: {candidate_results_path}")

    if len(candidates_df) == 0:
        print("\nNo strict candidate parameter set was found.")
        print("The script will still save the top-ranked parameter set.")
        best_row = results_df.iloc[0].to_dict()
    else:
        print(f"\nNumber of candidate parameter sets: {len(candidates_df)}")
        best_row = candidates_df.iloc[0].to_dict()

    #just the essentials
    best_params = {
        "full_dose": float(best_row["full_dose"]),
        "reduced_dose_fraction": float(best_row["reduced_dose_fraction"]),
        "reduced_dose": float(best_row["reduced_dose"]),
        "half_life": float(best_row["half_life"]),
        "baseline_growth_rate": float(best_row["baseline_growth_rate"]),
        "max_drug_effect": float(best_row["max_drug_effect"]),
        "ec50_ratio": float(best_row["ec50_ratio"]),
        "hill_coefficient": float(best_row["hill_coefficient"]),
        "score": float(best_row["score"]),
        "is_candidate": bool(best_row["is_candidate"]),
        "extinction_count": int(best_row["extinction_count"]),
        "saturation_count": int(best_row["saturation_count"]),
        "usable_signal_count": int(best_row["usable_signal_count"]),
        "high_fraction_range": float(best_row["high_fraction_range"]),
        "clustered_positive_count": int(best_row["clustered_positive_count"]),
        "mean_clustered_minus_dispersed": float(
            best_row["mean_clustered_minus_dispersed"]
        ),
    }

    print("\nSelected best parameter set:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    save_best_parameter_outputs(best_params)

    print("\nNext step:")
    print(
        "Open 'best_deterministic_population_summary.csv' and check whether "
        "the selected parameter set gives interpretable population dynamics. "
        "If it looks reasonable, use this parameter set as the baseline for "
        "the stochastic simulation stage."
    )


if __name__ == "__main__":
    main()
