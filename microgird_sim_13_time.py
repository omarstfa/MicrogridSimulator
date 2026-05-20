import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
#  SETTINGS – SCALE SIMULATION TIME, NOT FAILURE RATES
# ------------------------------------------------------------
EXPORT_FIGURES = False
SIM_YEARS = 50                     # very long simulation to collect failures
DT_HOURS = 1.0
N_STEPS = int(SIM_YEARS * 365 * 24 / DT_HOURS)
start_time = pd.Timestamp("2026-01-01 00:00:00")
np.random.seed(7)

# ------------------------------------------------------------
#  LOAD SOLAR DATA (only 5 years available, will be cycled)
# ------------------------------------------------------------
try:
    solar_df = pd.read_csv('solar_generation.csv', index_col='timestamp', parse_dates=True)
    print("Loaded solar generation data")
except FileNotFoundError:
    print("ERROR: solar_generation.csv not found. Please run solar_generation.py first.")
    exit()

# ------------------------------------------------------------
#  SYSTEM PARAMETERS
# ------------------------------------------------------------
BATTERY_CAPACITY_KWH = 5
BATTERY_MAX_CHARGE_RATE_KW = 2.5
BATTERY_MAX_DISCHARGE_RATE_KW = 2.5
BATTERY_EFFICIENCY = 0.95

BASE_LOAD_KW = 2.5
PEAK_LOAD_KW = 5
LOAD_VARIABILITY = 0.3

# ------------------------------------------------------------
#  COMPONENT DEFINITIONS – REALISTIC VALUES FROM TABLE 4 (NO SCALING)
# ------------------------------------------------------------
components = {
    "Grid": {
        "dist": "exponential",
        "lambda": 5.7e-5,          # original λ = 5.7e-5 /h
        "mu": 0.1,                 # original μ = 0.1 /h
        "up": True
    },
    "PCC_Breaker": {
        "dist": "exponential",
        "lambda": 1.0e-6,
        "mu": 0.5,
        "up": True
    },
    "PCC_Panel": {
        "dist": "exponential",
        "lambda": 2.0e-6,
        "mu": 0.25,
        "up": True
    },
    "PV_Array": {
        "dist": "weibull",
        "shape": 2.5,
        "scale": 200000.0,         # original η = 200,000 h
        "mu": 0.05,
        "up": True,
        "age": 0
    },
    "PV_Inverter": {
        "dist": "weibull",
        "shape": 2.2,
        "scale": 60000.0,
        "mu": 0.083,
        "up": True,
        "age": 0
    },
    "Battery_Pack": {
        "dist": "weibull",
        "shape": 3.0,
        "scale": 40000.0,
        "mu": 0.033,
        "up": True,
        "age": 0
    },
    "BMS": {
        "dist": "exponential",
        "lambda": 3.0e-6,
        "mu": 0.125,
        "up": True
    },
    "PCS": {
        "dist": "weibull",
        "shape": 2.2,
        "scale": 60000.0,
        "mu": 0.083,
        "up": True,
        "age": 0
    }
}

# ------------------------------------------------------------
#  HELPER FUNCTIONS (unchanged)
# ------------------------------------------------------------
def attempt_failure(comp):
    if comp.get("dist", "exponential") == "exponential":
        return np.random.rand() < comp["lambda"] * DT_HOURS
    elif comp["dist"] == "weibull":
        beta = comp["shape"]
        eta = comp["scale"]
        t1 = comp["age"]
        t2 = comp["age"] + DT_HOURS
        F_t1 = 1 - np.exp(-(t1/eta)**beta)
        F_t2 = 1 - np.exp(-(t2/eta)**beta)
        prob_failure = (F_t2 - F_t1) / (1 - F_t1 + 1e-10)
        return np.random.rand() < prob_failure
    return False

def attempt_repair(comp):
    return np.random.rand() < comp["mu"] * DT_HOURS

def calculate_load_demand(hour):
    daily_factor = np.sin(2 * np.pi * (hour % 24) / 24 - np.pi/2) * 0.5 + 0.8
    load = BASE_LOAD_KW * daily_factor + (PEAK_LOAD_KW - BASE_LOAD_KW) * np.random.random() * LOAD_VARIABILITY
    return max(1.0, load)

def calculate_fault_tree_events(comp_states):
    BE1 = not comp_states["Grid"]
    BE2 = not comp_states["PCC_Breaker"]
    BE3 = not comp_states["PCC_Panel"]
    BE4 = not comp_states["PV_Array"]
    BE5 = not comp_states["PV_Inverter"]
    BE6 = not comp_states["Battery_Pack"]
    BE7 = not comp_states["BMS"]
    BE8 = not comp_states["PCS"]
    grid_connection_failure = BE1 or BE2 or BE3
    pv_system_failure = BE4 or BE5
    battery_system_failure = BE6 or BE7 or BE8
    reduced_output_from_der = pv_system_failure or battery_system_failure
    return {
        "grid_connection_failure": grid_connection_failure,
        "pv_system_failure": pv_system_failure,
        "battery_system_failure": battery_system_failure,
        "reduced_output_from_der": reduced_output_from_der,
    }

# ------------------------------------------------------------
#  SIMULATION LOOP (long time, realistic parameters)
# ------------------------------------------------------------
fault_log = []
state_log = []

soc = 50.0
battery_kwh = soc / 100 * BATTERY_CAPACITY_KWH
prev_soc = soc
prev_pv_output = 0

# Pre‑compute solar data for repetition (cycle through the 5‑year data)
solar_years = 5
solar_hours = solar_years * 365 * 24
solar_array = solar_df['pv_generation_kw'].values

for step in range(N_STEPS):
    # Cyclic solar data: repeat the 5‑year pattern
    solar_idx = step % solar_hours
    solar_generation = solar_array[solar_idx]
    current_time = start_time + pd.Timedelta(hours=step * DT_HOURS)

    load_demand = calculate_load_demand(step)

    # Update component ages (Weibull)
    for name, comp in components.items():
        if comp["up"] and comp.get("dist") == "weibull":
            comp["age"] += DT_HOURS

    # Failures / repairs
    for name, comp in components.items():
        if comp["up"] and attempt_failure(comp):
            comp["up"] = False
            if comp.get("dist") == "weibull":
                comp["age"] = 0
            fault_log.append([current_time, name, "FAILURE"])
        elif not comp["up"] and attempt_repair(comp):
            comp["up"] = True
            if comp.get("dist") == "weibull":
                comp["age"] = 0
            fault_log.append([current_time, name, "REPAIR"])

    # Energy dispatch (unchanged logic)
    pv_available = components["PV_Array"]["up"] and components["PV_Inverter"]["up"]
    battery_hardware_ok = components["Battery_Pack"]["up"] and components["PCS"]["up"] and components["BMS"]["up"]
    grid_available = components["Grid"]["up"] and components["PCC_Breaker"]["up"] and components["PCC_Panel"]["up"]

    pv_power = solar_generation if (pv_available and solar_generation > 0.01) else 0
    battery_can_discharge = battery_hardware_ok and battery_kwh > 0.01
    battery_max_discharge = min(battery_kwh / DT_HOURS, BATTERY_MAX_DISCHARGE_RATE_KW) if battery_can_discharge else 0
    battery_can_charge_hw = battery_hardware_ok and battery_kwh < BATTERY_CAPACITY_KWH

    pv_to_load = 0
    battery_to_load = 0
    grid_to_load = 0
    pv_to_battery = 0
    remaining_load = load_demand

    if pv_power > 0:
        pv_to_load = min(pv_power, remaining_load)
        remaining_load -= pv_to_load
        pv_excess = pv_power - pv_to_load
    else:
        pv_excess = 0

    can_charge = (pv_excess > 0.01 and battery_can_charge_hw and components["PCC_Panel"]["up"])
    if can_charge:
        charge_power = min(pv_excess, BATTERY_MAX_CHARGE_RATE_KW)
        energy_to_add = charge_power * DT_HOURS * BATTERY_EFFICIENCY
        space_available = BATTERY_CAPACITY_KWH - battery_kwh
        energy_to_add = min(energy_to_add, space_available)
        pv_to_battery = energy_to_add / (DT_HOURS * BATTERY_EFFICIENCY)
        battery_kwh += energy_to_add

    if remaining_load > 0.01 and battery_can_discharge:
        battery_to_load = min(battery_max_discharge, remaining_load)
        energy_discharged = battery_to_load * DT_HOURS / BATTERY_EFFICIENCY
        energy_discharged = min(energy_discharged, battery_kwh)
        battery_to_load = energy_discharged * BATTERY_EFFICIENCY / DT_HOURS
        battery_kwh -= energy_discharged
        remaining_load -= battery_to_load

    if remaining_load > 0.01 and grid_available:
        grid_to_load = remaining_load
        remaining_load = 0

    loss_of_supply_actual = remaining_load > 0.01
    soc = (battery_kwh / BATTERY_CAPACITY_KWH) * 100

    # Operational logging
    if prev_soc > 0.1 and soc <= 0.1:
        fault_log.append([current_time, "Battery_Pack", "SOC_DEPLETED"])
    if prev_soc <= 0.1 and soc > 0.1:
        fault_log.append([current_time, "Battery_Pack", "SOC_RECOVERED"])
    if prev_pv_output > 0.01 and solar_generation <= 0.01:
        fault_log.append([current_time, "PV_Array", "OUTPUT_ZERO"])
    if prev_pv_output <= 0.01 and solar_generation > 0.01:
        fault_log.append([current_time, "PV_Array", "OUTPUT_ACTIVE"])

    prev_soc = soc
    prev_pv_output = solar_generation

    # Overridden states for fault tree
    component_states_for_ft = {name: comp["up"] for name, comp in components.items()}
    pv_operational = solar_generation > 0.01
    pv_sufficient = solar_generation >= load_demand
    component_states_for_ft["PV_Array"] = component_states_for_ft["PV_Array"] and pv_sufficient
    battery_pack_operational = soc > 0.1
    component_states_for_ft["Battery_Pack"] = component_states_for_ft["Battery_Pack"] and battery_pack_operational
    fault_tree_events = calculate_fault_tree_events(component_states_for_ft)

    row = {
        "loss_of_supply": int(loss_of_supply_actual),
        "grid_connection_failure": int(fault_tree_events["grid_connection_failure"]),
        "reduced_output_from_der": int(fault_tree_events["reduced_output_from_der"]),
        "soc": soc, "battery_kwh": battery_kwh,
        "solar_generation_kw": solar_generation, "load_demand_kw": load_demand,
        "pv_to_load_kw": pv_to_load, "pv_to_battery_kw": pv_to_battery,
        "battery_to_load_kw": battery_to_load, "grid_to_load_kw": grid_to_load,
        "unmet_load_kw": remaining_load, "can_charge": int(can_charge),
        "battery_can_discharge": int(battery_can_discharge),
        "pv_operational": int(pv_operational), "pv_sufficient": int(pv_sufficient),
        "battery_operational": int(battery_pack_operational)
    }
    for name in components.keys():
        row[name] = int(component_states_for_ft[name])
        row[name + "_hw"] = int(components[name]["up"])
    state_log.append(row)

    # Progress indicator
    if (step+1) % (365*24) == 0:
        print(f"Simulated { (step+1)//(365*24) } years / {SIM_YEARS} years")

# ------------------------------------------------------------
#  CREATE DATAFRAMES AND EXPORT
# ------------------------------------------------------------
full_df = pd.DataFrame(state_log)
full_df.index = pd.date_range(start=start_time, periods=len(full_df), freq="h")
fault_df = pd.DataFrame(fault_log, columns=["time", "component", "event"]).set_index("time")

state_columns = [
    "loss_of_supply", "grid_connection_failure", "reduced_output_from_der",
    "Grid", "PCC_Breaker", "PCC_Panel", "PV_Array", "PV_Inverter",
    "Battery_Pack", "BMS", "PCS"
]
state_df = full_df[state_columns].copy()
state_df.to_csv("PyFTE_Gird/state_data_with_override.csv")

dispatch_columns = [
    "soc", "battery_kwh", "solar_generation_kw", "load_demand_kw",
    "pv_to_load_kw", "pv_to_battery_kw", "battery_to_load_kw", "grid_to_load_kw",
    "unmet_load_kw", "can_charge"
]
dispatch_df = full_df[dispatch_columns].copy()
dispatch_df.to_csv("PyFTE_Gird/sensor_data.csv")
fault_df.to_csv("PyFTE_Gird/fault_log.csv")

print("\n" + "="*60)
print(f"Long‑time simulation finished: {SIM_YEARS} years")
print("Files written (realistic parameters, no scaling):")
print("  PyFTE_Gird/state_data_with_override.csv")
print("  PyFTE_Gird/sensor_data.csv")
print("  PyFTE_Gird/fault_log.csv")