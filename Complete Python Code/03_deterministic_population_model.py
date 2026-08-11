from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#paths for files
DATA_DIR = Path(".")
INPUT_FILE = DATA_DIR / "simulated_ciprofloxacin_exposure_trajectories.csv"

OUTPUT_DIR = DATA_DIR / "deterministic_model_output"
OUTPUT_DIR.mkdir(exist_ok=True)

#low mic phenotype is modal MIC, high mic phenotype is first MIC category on ECOFF
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

#baseline values, may be subject to change
INITIAL_TOTAL_POPULATION = 1e6       # cells
INITIAL_HIGH_FRACTION = 1e-4         # initial high-MIC phenotype fraction
CARRYING_CAPACITY = 1e9              # maximum population size, cells

BASELINE_GROWTH_RATE = 0.8           # h^-1
MAX_DRUG_EFFECT = 3.0                # h^-1
EC50_RATIO = 1.0                     # drug effect is half-maximal when C/MIC = 1
HILL_COEFFICIENT = 2.0               # steepness of drug response curve

HIGH_MIC_FITNESS_COST = 0.05         # 5% reduction in baseline growth rate
MIN_POPULATION = 1e-12               # lower bound to avoid numerical underflow

#funcs
def drug_effect(concentration: float, mic: float) -> float:
    if mic <= 0:
        raise ValueError("MIC must be greater than zero.")

    exposure_ratio = concentration / mic

    numerator = exposure_ratio ** HILL_COEFFICIENT
    denominator = (EC50_RATIO ** HILL_COEFFICIENT) + numerator

    if denominator == 0:
        return 0.0

    return MAX_DRUG_EFFECT * numerator / denominator


def update_population(
    n_low: float,
    n_high: float,
    concentration: float,
    low_mic: float,
    high_mic: float,
    dt: float,
) -> tuple[float, float, dict]:

    n_total = n_low + n_high
    logistic_factor = max(0.0, 1.0 - (n_total / CARRYING_CAPACITY))
    growth_low = BASELINE_GROWTH_RATE * logistic_factor
    growth_high = BASELINE_GROWTH_RATE * (1.0 - HIGH_MIC_FITNESS_COST) * logistic_factor

    #effects
    effect_low = drug_effect(concentration, low_mic)
    effect_high = drug_effect(concentration, high_mic)
    net_low = growth_low - effect_low
    net_high = growth_high - effect_high
    n_low_next = n_low * np.exp(net_low * dt)
    n_high_next = n_high * np.exp(net_high * dt)
    n_low_next = max(n_low_next, MIN_POPULATION)
    n_high_next = max(n_high_next, MIN_POPULATION)

    diagnostics = {
        "exposure_ratio_low": concentration / low_mic,
        "exposure_ratio_high": concentration / high_mic,
        "drug_effect_low": effect_low,
        "drug_effect_high": effect_high,
        "net_growth_low": net_low,
        "net_growth_high": net_high,
    }

    return n_low_next, n_high_next, diagnostics


def simulate_one_condition(
    trajectory_df: pd.DataFrame,
    species: str,
    scenario: str,
    low_mic: float,
    high_mic: float,
) -> pd.DataFrame:

    time_values = trajectory_df["time_h"].to_numpy()
    concentration_values = trajectory_df[scenario].to_numpy()

    #estimate dt
    dt_values = np.diff(time_values)
    if len(dt_values) == 0:
        raise ValueError("The trajectory file must contain more than one time point.")

    dt = float(np.median(dt_values))
    n_high = INITIAL_TOTAL_POPULATION * INITIAL_HIGH_FRACTION
    n_low = INITIAL_TOTAL_POPULATION - n_high

    rows = []

    for idx, time_h in enumerate(time_values):
        concentration = float(concentration_values[idx])

        n_total = n_low + n_high
        high_fraction = n_high / n_total if n_total > 0 else np.nan
        effect_low = drug_effect(concentration, low_mic)
        effect_high = drug_effect(concentration, high_mic)

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
            "exposure_ratio_low": concentration / low_mic,
            "exposure_ratio_high": concentration / high_mic,
            "drug_effect_low": effect_low,
            "drug_effect_high": effect_high,
        })

        #no update no no no no update
        if idx == len(time_values) - 1:
            break

        n_low, n_high, _ = update_population(
            n_low=n_low,
            n_high=n_high,
            concentration=concentration,
            low_mic=low_mic,
            high_mic=high_mic,
            dt=dt,
        )

    return pd.DataFrame(rows)

def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Place 'simulated_ciprofloxacin_exposure_trajectories.csv' "
            "in the same folder as this script."
        )

    trajectory_df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "time_h",
        "regular",
        "dispersed_troughs",
        "clustered_troughs",
    }

    missing_columns = required_columns - set(trajectory_df.columns)
    if missing_columns:
        raise ValueError(
            f"The input trajectory file is missing columns: {sorted(missing_columns)}"
        )

    trajectory_df["time_h"] = pd.to_numeric(trajectory_df["time_h"], errors="coerce")
    for col in ["regular", "dispersed_troughs", "clustered_troughs"]:
        trajectory_df[col] = pd.to_numeric(trajectory_df[col], errors="coerce")

    trajectory_df = trajectory_df.dropna(subset=list(required_columns)).copy()
    trajectory_df = trajectory_df.sort_values("time_h").reset_index(drop=True)

    scenarios = ["regular", "dispersed_troughs", "clustered_troughs"]

    all_timeseries = []

    for species, mic_info in REPRESENTATIVE_MIC.items():
        for scenario in scenarios:
            result = simulate_one_condition(
                trajectory_df=trajectory_df,
                species=species,
                scenario=scenario,
                low_mic=mic_info["low_mic"],
                high_mic=mic_info["high_mic"],
            )
            all_timeseries.append(result)

    timeseries_df = pd.concat(all_timeseries, ignore_index=True)

    #final summary
    summary_rows = []

    for (species, scenario), group in timeseries_df.groupby(["species", "scenario"]):
        group = group.sort_values("time_h")

        initial_row = group.iloc[0]
        final_row = group.iloc[-1]

        initial_high_fraction = float(initial_row["high_fraction"])
        final_high_fraction = float(final_row["high_fraction"])

        max_high_fraction = float(group["high_fraction"].max())
        time_max_high_fraction = float(
            group.loc[group["high_fraction"].idxmax(), "time_h"]
        )

        auc_0_72 = float(
            np.trapz(group["concentration_mg_l"], group["time_h"])
        )

        low_mic = float(final_row["low_mic"])
        high_mic = float(final_row["high_mic"])

        #window
        selective_window_mask = (
            (group["concentration_mg_l"] >= low_mic)
            & (group["concentration_mg_l"] < high_mic)
        )

        #time spent inbetween
        time_values = group["time_h"].to_numpy()
        if len(time_values) > 1:
            dt = float(np.median(np.diff(time_values)))
        else:
            dt = np.nan

        time_in_selective_window_h = float(selective_window_mask.sum() * dt)

        fold_change_high_fraction = (
            final_high_fraction / initial_high_fraction
            if initial_high_fraction > 0
            else np.nan
        )

        summary_rows.append({
            "species": species,
            "scenario": scenario,
            "low_mic": low_mic,
            "high_mic": high_mic,
            "initial_total_population": float(initial_row["N_total"]),
            "final_total_population": float(final_row["N_total"]),
            "initial_high_fraction": initial_high_fraction,
            "final_high_fraction": final_high_fraction,
            "fold_change_high_fraction": fold_change_high_fraction,
            "max_high_fraction": max_high_fraction,
            "time_max_high_fraction_h": time_max_high_fraction,
            "AUC_0_72": auc_0_72,
            "time_in_selective_window_h": time_in_selective_window_h,
        })

    summary_df = pd.DataFrame(summary_rows)

    #output save
    timeseries_path = OUTPUT_DIR / "deterministic_population_timeseries.csv"
    summary_path = OUTPUT_DIR / "deterministic_population_summary.csv"

    timeseries_df.to_csv(timeseries_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    scenario_label_map = {
        "regular": "Regular exposure",
        "dispersed_troughs": "Dispersed troughs",
        "clustered_troughs": "Clustered troughs",
    }

    species_filename_map = {
        "Escherichia coli": "e_coli",
        "Klebsiella pneumoniae": "k_pneumoniae",
        "Pseudomonas aeruginosa": "p_aeruginosa",
    }

    for species, mic_info in REPRESENTATIVE_MIC.items():
        species_df = timeseries_df[timeseries_df["species"] == species]

        plt.figure(figsize=(10, 6))

        for scenario in scenarios:
            plot_df = species_df[species_df["scenario"] == scenario]
            plt.plot(
                plot_df["time_h"],
                plot_df["high_fraction"],
                label=scenario_label_map[scenario],
            )

        plt.xlabel("Time (h)")
        plt.ylabel("High-MIC phenotype fraction")
        plt.title(
            f"Figure 3. Deterministic high-MIC fraction dynamics in "
            f"{mic_info['short_name']}"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        figure_path = OUTPUT_DIR / (
            f"figure3_{species_filename_map[species]}_"
            "deterministic_high_mic_fraction.png"
        )

        plt.savefig(figure_path, dpi=300)
        plt.close()

    #bar chart
    plot_summary = summary_df.copy()
    plot_summary["species_short"] = plot_summary["species"].map(
        {s: info["short_name"] for s, info in REPRESENTATIVE_MIC.items()}
    )
    plot_summary["scenario_label"] = plot_summary["scenario"].map(scenario_label_map)

    species_list = [info["short_name"] for info in REPRESENTATIVE_MIC.values()]
    x = np.arange(len(species_list))
    width = 0.25

    plt.figure(figsize=(10, 6))

    for i, scenario in enumerate(scenarios):
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
    plt.title("Figure 4. Final high-MIC phenotype fraction at 72 h")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    figure4_path = OUTPUT_DIR / "figure4_final_high_mic_fraction.png"
    plt.savefig(figure4_path, dpi=300)
    plt.close()

    #printing outputss
    print("\nDeterministic population model completed.")
    print(f"Timeseries saved to: {timeseries_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Figures saved to: {OUTPUT_DIR}")
    print("\nSummary preview:")
    print(summary_df)


if __name__ == "__main__":
    main()
