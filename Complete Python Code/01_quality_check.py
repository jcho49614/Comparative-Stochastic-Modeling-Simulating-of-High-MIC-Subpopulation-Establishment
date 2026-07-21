#!/usr/bin/env python3
"""
01_quality_check.py

Quality-control script for EUCAST ciprofloxacin MIC-distribution CSV files.

Expected input filenames:
    *_ciprofloxacin_eucast_mic.csv

Optional matching metadata filenames:
    *_ciprofloxacin_eucast_metadata.txt

The script checks:
1. MIC-bin count total vs. EUCAST-reported observations
2. Number of MIC categories
3. Whether ECOFF is present as an exact MIC bin
4. Missing, non-positive, or negative values
5. Whether normalized frequencies sum to 1

Outputs are written to:
    quality_check_output/
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# 1. File locations
# ============================================================

# Set this to the folder containing your MIC CSV files.
# "." means the same folder as this script.
DATA_DIR = Path(".")

# Folder for generated output files.
OUTPUT_DIR = DATA_DIR / "quality_check_output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Read metadata
# ============================================================

def read_metadata(metadata_path: Path) -> dict:
    """Read key values from a matching EUCAST metadata TXT file."""

    result = {
        "reported_observations": None,
        "metadata_ecoff": None,
        "reference_database_date": None,
    }

    if not metadata_path.exists():
        print(f"  [Warning] Metadata file not found: {metadata_path.name}")
        return result

    text = metadata_path.read_text(encoding="utf-8")

    match_obs = re.search(r"Reported observations:\s*([0-9,]+)", text)
    if match_obs:
        result["reported_observations"] = int(
            match_obs.group(1).replace(",", "")
        )

    match_ecoff = re.search(r"ECOFF:\s*([0-9.]+)\s*mg/L", text)
    if match_ecoff:
        result["metadata_ecoff"] = float(match_ecoff.group(1))

    match_date = re.search(
        r"Reference database date:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})",
        text,
    )
    if match_date:
        result["reference_database_date"] = match_date.group(1)

    return result


# ============================================================
# 3. Check one MIC CSV file
# ============================================================

def quality_check_mic_file(csv_path: Path) -> tuple[pd.DataFrame, dict]:
    """
    Read and check one EUCAST MIC CSV.

    Required columns:
        species
        antibiotic
        mic_mg_l
        isolate_count
        ecoff_mg_l
    """

    print("\n" + "=" * 72)
    print(f"Checking: {csv_path.name}")
    print("=" * 72)

    df = pd.read_csv(csv_path)

    required_columns = {
        "species",
        "antibiotic",
        "mic_mg_l",
        "isolate_count",
        "ecoff_mg_l",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{csv_path.name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Convert numeric columns safely.
    df["mic_mg_l"] = pd.to_numeric(df["mic_mg_l"], errors="coerce")
    df["isolate_count"] = pd.to_numeric(df["isolate_count"], errors="coerce")
    df["ecoff_mg_l"] = pd.to_numeric(df["ecoff_mg_l"], errors="coerce")

    # Locate matching metadata file.
    metadata_name = csv_path.name.replace("_mic.csv", "_metadata.txt")
    metadata_path = csv_path.parent / metadata_name
    metadata = read_metadata(metadata_path)

    # A. Missing values
    missing_mic = int(df["mic_mg_l"].isna().sum())
    missing_count = int(df["isolate_count"].isna().sum())
    missing_ecoff = int(df["ecoff_mg_l"].isna().sum())

    # B. Invalid numeric values
    nonpositive_mic = int((df["mic_mg_l"] <= 0).sum())
    negative_count = int((df["isolate_count"] < 0).sum())

    # C. Duplicate MIC bins
    duplicate_mic_bins = int(df["mic_mg_l"].duplicated().sum())

    # D. Total displayed MIC-bin count
    displayed_bin_total = int(df["isolate_count"].sum())

    reported_observations = metadata["reported_observations"]
    observation_difference = None
    observation_difference_percent = None

    if reported_observations is not None and reported_observations > 0:
        observation_difference = displayed_bin_total - reported_observations
        observation_difference_percent = (
            observation_difference / reported_observations * 100
        )

    # E. ECOFF consistency
    unique_ecoffs = df["ecoff_mg_l"].dropna().unique()

    if len(unique_ecoffs) == 1:
        ecoff = float(unique_ecoffs[0])
    else:
        ecoff = np.nan

    ecoff_in_mic_bins = False
    if not np.isnan(ecoff):
        ecoff_in_mic_bins = bool(
            np.isclose(
                df["mic_mg_l"],
                ecoff,
                rtol=0,
                atol=1e-10,
            ).any()
        )

    ecoff_matches_metadata = None
    if metadata["metadata_ecoff"] is not None and not np.isnan(ecoff):
        ecoff_matches_metadata = bool(
            np.isclose(
                ecoff,
                metadata["metadata_ecoff"],
                rtol=0,
                atol=1e-10,
            )
        )

    # F. Normalize the MIC-bin counts
    if displayed_bin_total <= 0:
        raise ValueError(
            f"{csv_path.name}: total isolate_count must be greater than zero."
        )

    df["normalized_frequency"] = (
        df["isolate_count"] / displayed_bin_total
    )
    normalized_frequency_sum = float(df["normalized_frequency"].sum())

    # G. Add phenotype group labels
    if np.isnan(ecoff):
        df["mic_group"] = "unknown_ecoff"
        low_mic_fraction = np.nan
        high_mic_fraction = np.nan
    else:
        df["mic_group"] = np.where(
            df["mic_mg_l"] <= ecoff,
            "wild_type_or_lower_MIC",
            "high_MIC_non_wild_type",
        )
        low_mic_fraction = float(
            df.loc[
                df["mic_mg_l"] <= ecoff,
                "normalized_frequency",
            ].sum()
        )
        high_mic_fraction = float(
            df.loc[
                df["mic_mg_l"] > ecoff,
                "normalized_frequency",
            ].sum()
        )

    species_values = df["species"].dropna().unique()
    antibiotic_values = df["antibiotic"].dropna().unique()

    summary = {
        "file_name": csv_path.name,
        "species": ", ".join(map(str, species_values)),
        "antibiotic": ", ".join(map(str, antibiotic_values)),
        "mic_category_count": len(df),
        "displayed_mic_bin_total": displayed_bin_total,
        "reported_observations": reported_observations,
        "difference_bin_total_minus_reported": observation_difference,
        "difference_percent": observation_difference_percent,
        "ecoff_mg_l": ecoff,
        "ecoff_in_mic_bins": ecoff_in_mic_bins,
        "ecoff_matches_metadata": ecoff_matches_metadata,
        "missing_mic_values": missing_mic,
        "missing_count_values": missing_count,
        "missing_ecoff_values": missing_ecoff,
        "nonpositive_mic_values": nonpositive_mic,
        "negative_count_values": negative_count,
        "duplicate_mic_bins": duplicate_mic_bins,
        "normalized_frequency_sum": normalized_frequency_sum,
        "wild_type_or_lower_mic_fraction": low_mic_fraction,
        "high_mic_fraction": high_mic_fraction,
    }

    # Print summary
    print(f"Species: {summary['species']}")
    print(f"Antibiotic: {summary['antibiotic']}")
    print(f"Number of MIC categories: {summary['mic_category_count']}")
    print(f"Displayed MIC-bin total: {summary['displayed_mic_bin_total']}")
    print(f"Reported observations: {summary['reported_observations']}")
    print(
        "Difference between displayed total and reported observations: "
        f"{summary['difference_bin_total_minus_reported']}"
    )
    print(f"ECOFF: {summary['ecoff_mg_l']} mg/L")
    print(f"ECOFF included as MIC bin: {summary['ecoff_in_mic_bins']}")
    print(
        f"Normalized frequency sum: "
        f"{summary['normalized_frequency_sum']:.12f}"
    )
    print(
        f"Wild-type / lower-MIC fraction: "
        f"{summary['wild_type_or_lower_mic_fraction']:.4f}"
    )
    print(
        f"High-MIC / non-wild-type fraction: "
        f"{summary['high_mic_fraction']:.4f}"
    )

    # Collect warnings
    warnings = []

    if missing_mic > 0:
        warnings.append(f"Missing MIC values: {missing_mic}")
    if missing_count > 0:
        warnings.append(f"Missing isolate counts: {missing_count}")
    if missing_ecoff > 0:
        warnings.append(f"Missing ECOFF values: {missing_ecoff}")
    if nonpositive_mic > 0:
        warnings.append(f"Non-positive MIC values: {nonpositive_mic}")
    if negative_count > 0:
        warnings.append(f"Negative isolate counts: {negative_count}")
    if duplicate_mic_bins > 0:
        warnings.append(f"Duplicate MIC bins: {duplicate_mic_bins}")
    if not ecoff_in_mic_bins:
        warnings.append("ECOFF is not included as an exact MIC bin.")
    if not np.isclose(normalized_frequency_sum, 1.0, atol=1e-12):
        warnings.append("Normalized frequency sum is not equal to 1.")
    if (
        observation_difference_percent is not None
        and abs(observation_difference_percent) > 1
    ):
        warnings.append(
            "Difference between displayed MIC-bin total and reported "
            "observations exceeds 1%."
        )

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\nQuality check passed without major warnings.")

    return df, summary


# ============================================================
# 4. Run checks for all matching CSV files
# ============================================================

def main() -> None:
    csv_files = sorted(DATA_DIR.glob("*_ciprofloxacin_eucast_mic.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "No files matching '*_ciprofloxacin_eucast_mic.csv' "
            "were found in the selected DATA_DIR."
        )

    all_summaries = []
    all_cleaned_data = []

    for csv_file in csv_files:
        cleaned_df, summary_dict = quality_check_mic_file(csv_file)

        normalized_name = csv_file.name.replace(
            ".csv",
            "_normalized.csv",
        )
        output_file = OUTPUT_DIR / normalized_name
        cleaned_df.to_csv(output_file, index=False)

        print(f"\nSaved normalized file: {output_file}")

        all_summaries.append(summary_dict)
        all_cleaned_data.append(cleaned_df)

    summary_df = pd.DataFrame(all_summaries)
    summary_path = OUTPUT_DIR / "mic_data_quality_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    combined_df = pd.concat(all_cleaned_data, ignore_index=True)
    combined_path = (
        OUTPUT_DIR / "combined_ciprofloxacin_mic_normalized.csv"
    )
    combined_df.to_csv(combined_path, index=False)

    print("\n" + "=" * 72)
    print("All quality checks completed.")
    print(f"Summary saved to: {summary_path}")
    print(f"Combined normalized data saved to: {combined_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
