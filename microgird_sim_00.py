import numpy as np
import pandas as pd

# ============================================================
# SIMULATION PARAMETERS
# ============================================================

SIM_YEARS = 2
HOURS_PER_YEAR = 365 * 24
DT = 1.0                               # hours
SIM_HOURS = SIM_YEARS * HOURS_PER_YEAR
N_STEPS = int(SIM_HOURS / DT)

np.random.seed(42)

# ============================================================
# COMPONENT DEFINITIONS
# ============================================================

# lambda = failure rate [1/hour]
# mu     = repair rate  [1/hour]

components = {
    "Grid": {
        "lambda": 0.02 / 24,
        "mu": 0.25 / 24,
        "up": True
    },
    "PCC_Breaker": {
        "lambda": 1e-4,
        "mu": 0.5,
        "up": True
    },
    "Islanding_Controller": {
        "lambda": 2e-4,
        "mu": 0.3,
        "up": True
    },
    "PV_Array": {
        "lambda": 1 / 50000,
        "mu": 0.05,
        "up": True
    },
    "PV_Inverter": {
        "lambda": 3e-4,
        "mu": 0.2,
        "up": True
    },
    "Battery_Pack": {
        "lambda": 1 / 40000,
        "mu": 0.05,
        "up": True
    },
    "BMS": {
        "lambda": 2e-4,
        "mu": 0.3,
        "up": True
    },
    "PCS": {
        "lambda": 2e-4,
        "mu": 0.3,
        "up": True
    }
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def attempt_failure(comp):
    """Bernoulli trial for component failure"""
    return np.random.rand() < comp["lambda"] * DT


def attempt_repair(comp):
    """Bernoulli trial for component repair"""
    return np.random.rand() < comp["mu"] * DT


def generate_sensors(components, soc):
    """Continuous sensor data correlated with failures"""
    grid_up = components["Grid"]["up"]
    pv_inv_up = components["PV_Inverter"]["up"]
    battery_up = components["Battery_Pack"]["up"]

    return {
        "grid_voltage": (
            np.random.normal(400, 5)
            if grid_up else np.random.normal(0, 2)
        ),
        "pv_inverter_temp": (
            40 + (0 if pv_inv_up else 25) + np.random.normal(0, 2)
        ),
        "battery_temp": (
            30 + (0 if battery_up else 20) + np.random.normal(0, 1.5)
        ),
        "soc": soc + np.random.normal(0, 0.5),
    }


# ============================================================
# GROUND-TRUTH FAULT TREE (HIDDEN FROM LEARNING)
# ============================================================

def loss_of_supply(components):
    """
    Implements the ground-truth FT logic:

    TE = IE1 OR IE3

    IE1 = Grid_Outage AND (PCC_Failure OR Islanding_Controller_Failure)
    IE3 = Grid_Outage AND (PV_Unavailable AND BESS_Unavailable)
    """

    BE1 = not components["Grid"]["up"]
    BE2 = not components["PCC_Breaker"]["up"]
    BE3 = not components["Islanding_Controller"]["up"]
    BE4 = not components["PV_Array"]["up"]
    BE5 = not components["PV_Inverter"]["up"]
    BE6 = not components["Battery_Pack"]["up"]
    BE7 = not components["BMS"]["up"]
    BE8 = not components["PCS"]["up"]

    islanding_fails = BE2 or BE3
    immediate_failure = BE1 and islanding_fails

    pv_unavailable = BE4 or BE5
    bess_unavailable = BE6 or BE7 or BE8
    der_cannot_meet_load = pv_unavailable and bess_unavailable

    islanded_failure = BE1 and der_cannot_meet_load

    return immediate_failure or islanded_failure


# ============================================================
# DATA STORAGE
# ============================================================

fault_log = []        # event-driven
sensor_log = []       # time-series

soc = 80.0            # initial state of charge (%)

# ============================================================
# MAIN SIMULATION LOOP
# ============================================================

for step in range(N_STEPS):
    time = step * DT

    # --- failure / repair events ---
    for name, comp in components.items():
        if comp["up"] and attempt_failure(comp):
            comp["up"] = False
            fault_log.append([time, name, "FAILURE"])

        elif not comp["up"] and attempt_repair(comp):
            comp["up"] = True
            fault_log.append([time, name, "REPAIR"])

    # --- system-level effect on SoC ---
    if loss_of_supply(components):
        soc -= 2.0 * DT
    else:
        soc -= 0.1 * DT
    soc = np.clip(soc, 0, 100)

    # --- sensor readings ---
    sensors = generate_sensors(components, soc)

    # --- log snapshot ---
    row = {
        "time": time,
        "loss_of_supply": int(loss_of_supply(components))
    }

    for name, comp in components.items():
        row[f"{name}_up"] = int(comp["up"])

    row.update(sensors)
    sensor_log.append(row)

# ============================================================
# EXPORT DATASETS
# ============================================================

fault_df = pd.DataFrame(
    fault_log,
    columns=["time", "component", "event"]
)

sensor_df = pd.DataFrame(sensor_log)

fault_df.to_csv("fault_log.csv", index=False)
sensor_df.to_csv("sensor_data.csv", index=False)

# ============================================================
# SUMMARY
# ============================================================

print("Simulation complete.")
print(f"Simulated time: {SIM_YEARS} years")
print(f"Fault events:   {len(fault_df)}")
print(f"Sensor samples:{len(sensor_df)}")
print("\nFiles written:")
print(" - fault_log.csv")
print(" - sensor_data.csv")
