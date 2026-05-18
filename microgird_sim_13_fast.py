import numpy as np
import pandas as pd
import numba
from numba import njit
import time

# ------------------------------------------------------------
# SETTINGS – realistic, safe simulation length (100 years)
# ------------------------------------------------------------
SIM_YEARS = 100               # safe for datetime (2026 → 2126)
DT_HOURS = 1.0
N_STEPS = int(SIM_YEARS * 365 * 24 / DT_HOURS)
np.random.seed(7)

# ------------------------------------------------------------
# LOAD SOLAR DATA (pre‑extract as numpy array)
# ------------------------------------------------------------
try:
    solar_df = pd.read_csv('solar_generation.csv', index_col='timestamp', parse_dates=True)
    print("Loaded solar generation data")
except FileNotFoundError:
    print("ERROR: solar_generation.csv not found.")
    exit()

solar_array = solar_df['pv_generation_kw'].values   # shape (5 years * 8760,)
solar_years = 5
solar_hours = solar_years * 365 * 24

# ------------------------------------------------------------
# SYSTEM PARAMETERS (constants)
# ------------------------------------------------------------
BATTERY_CAPACITY_KWH = 5.0
BATTERY_MAX_CHARGE_RATE_KW = 2.5
BATTERY_MAX_DISCHARGE_RATE_KW = 2.5
BATTERY_EFFICIENCY = 0.95

BASE_LOAD_KW = 2.5
PEAK_LOAD_KW = 5.0
LOAD_VARIABILITY = 0.3

# ------------------------------------------------------------
# COMPONENT PARAMETERS (as arrays for Numba)
# ------------------------------------------------------------
comp_names = ["Grid", "PCC_Breaker", "PCC_Panel", "PV_Array", "PV_Inverter",
              "Battery_Pack", "BMS", "PCS"]
n_comp = len(comp_names)

# Exponential: 1, Weibull: 0
comp_is_exp = np.array([1, 1, 1, 0, 0, 0, 1, 0], dtype=np.int8)

# Exponential lambda & mu
comp_lambda = np.array([5.7e-5, 1.0e-6, 2.0e-6, 0.0, 0.0, 0.0, 3.0e-6, 0.0])
comp_mu     = np.array([0.1, 0.5, 0.25, 0.05, 0.083, 0.033, 0.125, 0.083])

# Weibull shape & scale
comp_weibull_shape = np.array([0.0, 0.0, 0.0, 2.5, 2.2, 3.0, 0.0, 2.2])
comp_weibull_scale = np.array([0.0, 0.0, 0.0, 200000.0, 60000.0, 40000.0, 0.0, 60000.0])

# Initial states
comp_up = np.ones(n_comp, dtype=np.int8)
comp_age = np.zeros(n_comp, dtype=np.float64)

# ------------------------------------------------------------
# LOAD DEMAND (precomputed)
# ------------------------------------------------------------
def precompute_load_demand(n_steps):
    hour_of_day = np.arange(n_steps) % 24
    daily_factor = np.sin(2 * np.pi * hour_of_day / 24 - np.pi/2) * 0.5 + 0.8
    random_part = np.random.rand(n_steps) * LOAD_VARIABILITY
    load = BASE_LOAD_KW * daily_factor + (PEAK_LOAD_KW - BASE_LOAD_KW) * random_part
    return np.maximum(1.0, load)

load_demand = precompute_load_demand(N_STEPS)

# ------------------------------------------------------------
# PRE‑GENERATE ALL RANDOM NUMBERS (for speed)
# ------------------------------------------------------------
rand_fail = np.random.rand(n_comp, N_STEPS)
rand_repair = np.random.rand(n_comp, N_STEPS)

# ------------------------------------------------------------
# Numba‑accelerated simulation kernel
# ------------------------------------------------------------
@njit
def simulate_one_step(step, comp_up, comp_age, battery_kwh, solar_power, load,
                      rand_fail_this, rand_repair_this):
    """
    Simulate one time step (1 hour) and return updated states.
    """
    # 1. Update ages of Weibull components that are up
    for i in range(n_comp):
        if comp_up[i] and comp_is_exp[i] == 0:
            comp_age[i] += DT_HOURS

    # 2. Failure / repair events
    for i in range(n_comp):
        if comp_up[i]:
            # attempt failure
            if comp_is_exp[i]:
                prob = comp_lambda[i] * DT_HOURS
                if rand_fail_this[i] < prob:
                    comp_up[i] = 0
                    if comp_is_exp[i] == 0:
                        comp_age[i] = 0.0
            else:  # Weibull
                beta = comp_weibull_shape[i]
                eta = comp_weibull_scale[i]
                t1 = comp_age[i]
                t2 = t1 + DT_HOURS
                F_t1 = 1.0 - np.exp(-(t1/eta)**beta)
                F_t2 = 1.0 - np.exp(-(t2/eta)**beta)
                prob = (F_t2 - F_t1) / (1.0 - F_t1 + 1e-12)
                if rand_fail_this[i] < prob:
                    comp_up[i] = 0
                    comp_age[i] = 0.0
        else:
            # attempt repair
            prob = comp_mu[i] * DT_HOURS
            if rand_repair_this[i] < prob:
                comp_up[i] = 1
                if comp_is_exp[i] == 0:
                    comp_age[i] = 0.0

    # 3. Energy dispatch
    pv_available = (comp_up[3] == 1) and (comp_up[4] == 1)
    battery_hw_ok = (comp_up[5] == 1) and (comp_up[7] == 1) and (comp_up[6] == 1)
    grid_available = (comp_up[0] == 1) and (comp_up[1] == 1) and (comp_up[2] == 1)

    pv_power = solar_power if (pv_available and solar_power > 0.01) else 0.0

    battery_can_discharge = battery_hw_ok and (battery_kwh > 0.01)
    max_discharge = min(battery_kwh / DT_HOURS, BATTERY_MAX_DISCHARGE_RATE_KW) if battery_can_discharge else 0.0

    remaining = load
    pv_to_load = 0.0
    pv_excess = 0.0
    if pv_power > 0:
        pv_to_load = min(pv_power, remaining)
        remaining -= pv_to_load
        pv_excess = pv_power - pv_to_load

    can_charge = (pv_excess > 0.01) and battery_hw_ok and (battery_kwh < BATTERY_CAPACITY_KWH) and (comp_up[2] == 1)
    if can_charge:
        charge_power = min(pv_excess, BATTERY_MAX_CHARGE_RATE_KW)
        energy_to_add = charge_power * DT_HOURS * BATTERY_EFFICIENCY
        space = BATTERY_CAPACITY_KWH - battery_kwh
        energy_to_add = min(energy_to_add, space)
        battery_kwh += energy_to_add

    battery_to_load = 0.0
    if remaining > 0.01 and battery_can_discharge:
        battery_to_load = min(max_discharge, remaining)
        energy_discharged = battery_to_load * DT_HOURS / BATTERY_EFFICIENCY
        energy_discharged = min(energy_discharged, battery_kwh)
        battery_to_load = energy_discharged * BATTERY_EFFICIENCY / DT_HOURS
        battery_kwh -= energy_discharged
        remaining -= battery_to_load

    grid_to_load = 0.0
    if remaining > 0.01 and grid_available:
        grid_to_load = remaining
        remaining = 0.0

    loss = (remaining > 0.01)

    return battery_kwh, loss, comp_up.copy(), comp_age.copy()

# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
print(f"Simulating {SIM_YEARS} years ({N_STEPS} hours) with Numba...")
start_time_total = time.time()

fault_log = []       # list of [step, component, event]
state_log = []       # list of dicts

battery_kwh = 50.0 / 100.0 * BATTERY_CAPACITY_KWH
prev_soc = 50.0
prev_pv = 0.0

print_interval = max(1, N_STEPS // 20)

for step in range(N_STEPS):
    solar_idx = step % solar_hours
    solar_power = solar_array[solar_idx]
    load = load_demand[step]

    rand_fail_step = rand_fail[:, step]
    rand_repair_step = rand_repair[:, step]

    new_battery_kwh, loss, new_up, new_age = simulate_one_step(
        step, comp_up, comp_age, battery_kwh, solar_power, load,
        rand_fail_step, rand_repair_step
    )

    # Log failures / repairs
    for i in range(n_comp):
        if comp_up[i] == 1 and new_up[i] == 0:
            fault_log.append([step, comp_names[i], "FAILURE"])
        elif comp_up[i] == 0 and new_up[i] == 1:
            fault_log.append([step, comp_names[i], "REPAIR"])

    comp_up = new_up
    comp_age = new_age
    battery_kwh = new_battery_kwh
    soc = battery_kwh / BATTERY_CAPACITY_KWH * 100.0

    # Operational events
    if prev_soc > 0.1 and soc <= 0.1:
        fault_log.append([step, "Battery_Pack", "SOC_DEPLETED"])
    if prev_soc <= 0.1 and soc > 0.1:
        fault_log.append([step, "Battery_Pack", "SOC_RECOVERED"])
    if prev_pv > 0.01 and solar_power <= 0.01:
        fault_log.append([step, "PV_Array", "OUTPUT_ZERO"])
    if prev_pv <= 0.01 and solar_power > 0.01:
        fault_log.append([step, "PV_Array", "OUTPUT_ACTIVE"])

    prev_soc = soc
    prev_pv = solar_power

    # Overridden states
    pv_operational = solar_power > 0.01
    pv_sufficient = solar_power >= load
    pv_override = (comp_up[3] == 1) and pv_sufficient
    battery_operational = soc > 0.1
    battery_override = (comp_up[5] == 1) and battery_operational

    row = {
        "loss_of_supply": 1 if loss else 0,
        "soc": soc,
        "battery_kwh": battery_kwh,
        "solar_generation_kw": solar_power,
        "load_demand_kw": load,
        "pv_operational": 1 if pv_operational else 0,
        "pv_sufficient": 1 if pv_sufficient else 0,
        "battery_operational": 1 if battery_operational else 0,
    }
    for i, name in enumerate(comp_names):
        if name == "PV_Array":
            row[name] = 1 if pv_override else 0
        elif name == "Battery_Pack":
            row[name] = 1 if battery_override else 0
        else:
            row[name] = comp_up[i]
        row[name + "_hw"] = comp_up[i]

    state_log.append(row)

    if (step+1) % print_interval == 0:
        pct = (step+1)/N_STEPS*100
        print(f"  Progress: {pct:.1f}% ({step+1}/{N_STEPS} steps)")

sim_time = time.time() - start_time_total
print(f"Simulation finished in {sim_time:.2f} seconds.")

# ------------------------------------------------------------
# CREATE DATAFRAMES AND EXPORT
# ------------------------------------------------------------
full_df = pd.DataFrame(state_log)
# Use integer index (step number) – no datetime overflow
full_df.index = range(len(full_df))

# Convert step numbers to datetime for fault log (optional, but keep safe)
start_datetime = pd.Timestamp("2026-01-01 00:00:00")
fault_df = pd.DataFrame(fault_log, columns=["step", "component", "event"])
fault_df["time"] = start_datetime + pd.to_timedelta(fault_df["step"], unit="h")
fault_df = fault_df.set_index("time").drop(columns="step")

state_columns = ["loss_of_supply"] + comp_names
state_df = full_df[state_columns].copy()
state_df.to_csv("PyFTE_Gird/state_data_with_override.csv")

dispatch_columns = ["soc", "battery_kwh", "solar_generation_kw", "load_demand_kw",
                    "pv_operational", "pv_sufficient", "battery_operational"]
dispatch_df = full_df[dispatch_columns].copy()
dispatch_df.to_csv("PyFTE_Gird/sensor_data.csv")

fault_df.to_csv("PyFTE_Gird/fault_log.csv")

print("\n" + "="*60)
print(f"Long‑time simulation finished: {SIM_YEARS} years")
print("Files written (realistic parameters, optimised with Numba):")
print("  PyFTE_Gird/state_data_with_override.csv")
print("  PyFTE_Gird/sensor_data.csv")
print("  PyFTE_Gird/fault_log.csv")