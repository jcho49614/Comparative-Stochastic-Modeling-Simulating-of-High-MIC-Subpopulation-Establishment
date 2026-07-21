#!/usr/bin/env python3
"""
02_generate_exposure_trajectories.py

Generate simulated ciprofloxacin exposure trajectories using a pulse-decay model.

Outputs:
    simulated_ciprofloxacin_exposure_trajectories.csv
    simulated_ciprofloxacin_exposure_metrics.csv
    figure2_simulated_ciprofloxacin_exposure_trajectories.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Basic settings
# ============================================================

total_time = 72          # total simulation time, h
dt = 0.1                 # time step, h
half_life = 4            # concentration half-life, h

full_dose = 4            # full-dose pulse, mg/L
reduced_dose = 1         # reduced-dose pulse, mg/L

dose_times = [0, 12, 24, 36, 48, 60]

# Concentration decay constant
k = np.log(2) / half_life

# Time array
time_points = np.round(np.arange(0, total_time + dt, dt), 1)


# ============================================================
# 2. Exposure scenario definitions
# ============================================================

dose_patterns = {
    "regular": [4, 4, 4, 4, 4, 4],
    "dispersed_troughs": [4, 1, 4, 4, 1, 4],
    "clustered_troughs": [4, 4, 1, 1, 4, 4],
}


# ============================================================
# 3. Pulse-decay model function
# ============================================================

def generate_concentration_trajectory(dose_pattern):
    """
    Generate the 0-72 h ciprofloxacin concentration trajectory
    according to the given dose pattern.
    """

    # Match dose times with dose values
    dose_dict = dict(zip(dose_times, dose_pattern))

    concentrations = []
    C = 0.0

    for t in time_points:
        # Add dose pulse if the current time is a dosing time
        if t in dose_dict:
            C += dose_dict[t]

        # Save current concentration
        concentrations.append(C)

        # Apply exponential decay before the next time step
        C = C * np.exp(-k * dt)

    return np.array(concentrations)


# ============================================================
# 4. Generate concentration trajectories for all scenarios
# ============================================================

trajectory_data = pd.DataFrame({
    "time_h": time_points
})

for scenario, pattern in dose_patterns.items():
    trajectory_data[scenario] = generate_concentration_trajectory(pattern)


# ============================================================
# 5. Calculate exposure metrics
# ============================================================

auc_results = []

for scenario in dose_patterns.keys():
    auc = np.trapz(
        trajectory_data[scenario],
        trajectory_data["time_h"]
    )

    auc_results.append({
        "scenario": scenario,
        "AUC_0_72": auc,
        "max_concentration": trajectory_data[scenario].max(),
        "min_concentration": trajectory_data[scenario].min(),
    })

auc_df = pd.DataFrame(auc_results)


# ============================================================
# 6. Save CSV files
# ============================================================

trajectory_data.to_csv(
    "simulated_ciprofloxacin_exposure_trajectories.csv",
    index=False
)

auc_df.to_csv(
    "simulated_ciprofloxacin_exposure_metrics.csv",
    index=False
)


# ============================================================
# 7. Draw and save Figure 2
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    trajectory_data["time_h"],
    trajectory_data["regular"],
    label="Regular exposure"
)

plt.plot(
    trajectory_data["time_h"],
    trajectory_data["dispersed_troughs"],
    label="Dispersed troughs"
)

plt.plot(
    trajectory_data["time_h"],
    trajectory_data["clustered_troughs"],
    label="Clustered troughs"
)

plt.xlabel("Time (h)")
plt.ylabel("Ciprofloxacin concentration (mg/L)")
plt.title("Figure 2. Simulated ciprofloxacin exposure trajectories")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "figure2_simulated_ciprofloxacin_exposure_trajectories.png",
    dpi=300
)

plt.show()


# ============================================================
# 8. Print results
# ============================================================

print("Exposure trajectories saved.")
print("Exposure metrics saved.")
print()
print(auc_df)
