import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
#  SETTINGS
# ------------------------------------------------------------
EXPORT_FIGURES = False          # keep export code, but do not save now
SIM_DAYS = 180
DT_HOURS = 1.0
N_STEPS = int((SIM_DAYS * 24) / DT_HOURS)
start_time = pd.Timestamp("2026-01-01 00:00:00")
np.random.seed(7)

# ------------------------------------------------------------
#  LOAD SOLAR DATA
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
BATTERY_CAPACITY_KWH = 20
BATTERY_MAX_CHARGE_RATE_KW = 5
BATTERY_MAX_DISCHARGE_RATE_KW = 5
BATTERY_EFFICIENCY = 0.95
PV_CAPACITY_KW = 10

BASE_LOAD_KW = 3
PEAK_LOAD_KW = 8
LOAD_VARIABILITY = 0.3

# ------------------------------------------------------------
#  COMPONENT DEFINITIONS
# ------------------------------------------------------------
components = {
    "Grid":                 {"lambda": 0.08 / 24, "mu": 0.4 / 24,  "up": True},
    "PCC_Breaker":          {"lambda": 5e-3,       "mu": 0.3,        "up": True},
    "Islanding_Controller": {"lambda": 7e-3,       "mu": 0.25,       "up": True},
    "PV_Array":             {"lambda": 5e-3,       "mu": 0.05,       "up": True},
    "PV_Inverter":          {"lambda": 8e-3,       "mu": 0.15,       "up": True},
    "Battery_Pack":         {"lambda": 6e-3,       "mu": 0.05,       "up": True},
    "BMS":                  {"lambda": 7e-3,       "mu": 0.25,       "up": True},
    "PCS":                  {"lambda": 7e-3,       "mu": 0.25,       "up": True}
}
 
# ------------------------------------------------------------
#  HELPER FUNCTIONS
# ------------------------------------------------------------
def attempt_failure(comp):
    return np.random.rand() < comp["lambda"] * DT_HOURS

def attempt_repair(comp):
    return np.random.rand() < comp["mu"] * DT_HOURS

def calculate_load_demand(hour):
    daily_factor = np.sin(2 * np.pi * (hour % 24) / 24 - np.pi/2) * 0.5 + 0.8
    load = BASE_LOAD_KW * daily_factor + (PEAK_LOAD_KW - BASE_LOAD_KW) * np.random.random() * LOAD_VARIABILITY
    return max(1.0, load)

def calculate_fault_tree_events(comp_states):
    BE1 = not comp_states["Grid"]
    BE2 = not comp_states["PCC_Breaker"]
    BE3 = not comp_states["Islanding_Controller"]
    BE4 = not comp_states["PV_Array"]
    BE5 = not comp_states["PV_Inverter"]
    BE6 = not comp_states["Battery_Pack"]
    BE7 = not comp_states["BMS"]
    BE8 = not comp_states["PCS"]
    immediate_failure = BE1 and (BE2 or BE3)
    islanded_failure = BE1 and ((BE4 or BE5) and (BE6 or BE7 or BE8))
    return {"immediate_failure": immediate_failure, "islanded_failure": islanded_failure}

# ------------------------------------------------------------
#  SIMULATION LOOP
# ------------------------------------------------------------
fault_log = []
state_log = []

soc = 50.0
battery_kwh = soc / 100 * BATTERY_CAPACITY_KWH

prev_soc = soc
prev_pv_output = 0

for step in range(N_STEPS):
    current_time = start_time + pd.Timedelta(hours=step * DT_HOURS)

    solar_generation = solar_df.loc[current_time, 'pv_generation_kw']
    load_demand = calculate_load_demand(step)

    # ---- hardware failures / repairs ----
    for name, comp in components.items():
        if comp["up"] and attempt_failure(comp):
            comp["up"] = False
            fault_log.append([current_time, name, "FAILURE"])
        elif not comp["up"] and attempt_repair(comp):
            comp["up"] = True
            fault_log.append([current_time, name, "REPAIR"])

    # ---- energy dispatch (uses hardware states) ----
    pv_available = components["PV_Array"]["up"] and components["PV_Inverter"]["up"]
    battery_hardware_ok = components["Battery_Pack"]["up"] and components["PCS"]["up"] and components["BMS"]["up"]
    grid_available = components["Grid"]["up"] and components["PCC_Breaker"]["up"] and components["Islanding_Controller"]["up"]

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

    can_charge = (pv_excess > 0.01 and
                  battery_can_charge_hw and
                  components["Islanding_Controller"]["up"])

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

    # ---- operational event logging ----
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

    # ---- overridden states for fault tree ----
    component_states_for_ft = {name: comp["up"] for name, comp in components.items()}
    pv_array_operational = solar_generation > 0.01
    component_states_for_ft["PV_Array"] = component_states_for_ft["PV_Array"] and pv_array_operational
    battery_pack_operational = soc > 0.1
    component_states_for_ft["Battery_Pack"] = component_states_for_ft["Battery_Pack"] and battery_pack_operational

    fault_tree_events = calculate_fault_tree_events(component_states_for_ft)

    # ---- store row ----
    row = {
        "loss_of_supply": int(loss_of_supply_actual),
        "immediate_failure": int(fault_tree_events["immediate_failure"]),
        "islanded_failure": int(fault_tree_events["islanded_failure"]),
        "soc": soc,
        "battery_kwh": battery_kwh,
        "solar_generation_kw": solar_generation,
        "load_demand_kw": load_demand,
        "pv_to_load_kw": pv_to_load,
        "pv_to_battery_kw": pv_to_battery,
        "battery_to_load_kw": battery_to_load,
        "grid_to_load_kw": grid_to_load,
        "unmet_load_kw": remaining_load,
        "can_charge": int(can_charge),
        "battery_can_discharge": int(battery_can_discharge),
        "pv_operational": int(pv_array_operational),
        "battery_operational": int(battery_pack_operational)
    }

    for name in components.keys():
        row[name] = int(component_states_for_ft[name])
    for name in components.keys():
        row[name + "_hw"] = int(components[name]["up"])

    state_log.append(row)

# ------------------------------------------------------------
#  CREATE DATAFRAMES
# ------------------------------------------------------------
full_df = pd.DataFrame(state_log)
full_df.index = pd.date_range(start=start_time, periods=len(full_df), freq="h")
fault_df = pd.DataFrame(fault_log, columns=["time", "component", "event"]).set_index("time")

# ------------------------------------------------------------
#  EXPORT DATA – TWO SEPARATE CSV FILES (EXACT COLUMNS)
# ------------------------------------------------------------
state_columns = [
    "loss_of_supply", "immediate_failure", "islanded_failure",
    "Grid", "PCC_Breaker", "Islanding_Controller",
    "PV_Array", "PV_Inverter", "Battery_Pack", "BMS", "PCS"
]
state_df = full_df[state_columns].copy()
state_df.to_csv("state_data_with_override.csv")
print("✓ state_data_with_override.csv saved")

dispatch_columns = [
    "soc", "battery_kwh",
    "solar_generation_kw", "load_demand_kw",
    "pv_to_load_kw", "pv_to_battery_kw",
    "battery_to_load_kw", "grid_to_load_kw",
    "unmet_load_kw", "can_charge"
]
dispatch_df = full_df[dispatch_columns].copy()
dispatch_df.to_csv("sensor_data.csv")
print("✓ sensor_data.csv saved")
fault_df.to_csv("fault_log.csv")
print("✓ fault_log.csv saved")

# ------------------------------------------------------------
#  SUMMARY STATISTICS
# ------------------------------------------------------------
print("\n" + "="*60)
print("SIMULATION SUMMARY")
print("="*60)
print(f"Simulation period: {SIM_DAYS} days ({N_STEPS} hours)")
print(f"\nFAULT TREE ANALYSIS (Based on Overridden States):")
print(f"  Immediate failures: {full_df['immediate_failure'].sum()} hours ({full_df['immediate_failure'].sum() / N_STEPS * 100:.2f}%)")
print(f"  Islanded failures: {full_df['islanded_failure'].sum()} hours ({full_df['islanded_failure'].sum() / N_STEPS * 100:.2f}%)")
print(f"\nACTUAL SYSTEM PERFORMANCE:")
print(f"  Loss of supply: {full_df['loss_of_supply'].sum()} hours ({full_df['loss_of_supply'].sum() / N_STEPS * 100:.2f}%)")
print(f"\nOVERRIDE STATISTICS:")
print(f"  Hours PV_Array marked down (operational): {(full_df['PV_Array'] != full_df['PV_Array_hw']).sum()}")
print(f"  Hours Battery_Pack marked down (operational): {(full_df['Battery_Pack'] != full_df['Battery_Pack_hw']).sum()}")
print(f"  Total PV output zero events: {(full_df['pv_operational'] == 0).sum()} hours ({(full_df['pv_operational'] == 0).sum() / N_STEPS * 100:.1f}%)")
print(f"  Total battery depleted events: {(full_df['battery_operational'] == 0).sum()} hours ({(full_df['battery_operational'] == 0).sum() / N_STEPS * 100:.1f}%)")
print(f"\nENERGY STATISTICS:")
print(f"  Total solar generation: {full_df['solar_generation_kw'].sum():.1f} kWh")
print(f"  Total load demand: {full_df['load_demand_kw'].sum():.1f} kWh")
print(f"  Total unmet load: {full_df['unmet_load_kw'].sum():.1f} kWh")
print(f"\nENERGY DISPATCH:")
print(f"  PV to load: {full_df['pv_to_load_kw'].sum():.1f} kWh ({full_df['pv_to_load_kw'].sum() / full_df['load_demand_kw'].sum() * 100:.1f}%)")
print(f"  PV to battery: {full_df['pv_to_battery_kw'].sum():.1f} kWh")
print(f"  Battery to load: {full_df['battery_to_load_kw'].sum():.1f} kWh ({full_df['battery_to_load_kw'].sum() / full_df['load_demand_kw'].sum() * 100:.1f}%)")
print(f"  Grid to load: {full_df['grid_to_load_kw'].sum():.1f} kWh ({full_df['grid_to_load_kw'].sum() / full_df['load_demand_kw'].sum() * 100:.1f}%)")
print(f"\nBATTERY STATISTICS:")
print(f"  Final SOC: {soc:.1f}% ({battery_kwh:.1f} kWh)")
print(f"  Charging hours: {(full_df['pv_to_battery_kw'] > 0).sum()}")
print(f"  Discharging hours: {(full_df['battery_to_load_kw'] > 0).sum()}")
print(f"  Average SOC: {full_df['soc'].mean():.1f}%")
print(f"  Min SOC: {full_df['soc'].min():.1f}%")
print(f"  Max SOC: {full_df['soc'].max():.1f}%")
print(f"\nHARDWARE FAILURES (Random Events):")
failure_counts = fault_df[fault_df['event'] == 'FAILURE'].groupby('component').size()
repair_counts = fault_df[fault_df['event'] == 'REPAIR'].groupby('component').size()
for comp in components.keys():
    failures = failure_counts.get(comp, 0)
    repairs = repair_counts.get(comp, 0)
    if failures > 0 or repairs > 0:
        print(f"  {comp}: {failures} failures, {repairs} repairs")
print(f"\nOPERATIONAL EVENTS (Not Hardware Failures):")
soc_depleted = len(fault_df[fault_df['event'] == 'SOC_DEPLETED'])
pv_zero_events = len(fault_df[fault_df['event'] == 'OUTPUT_ZERO'])
print(f"  Battery SOC depleted: {soc_depleted} times")
print(f"  Battery SOC recovered: {len(fault_df[fault_df['event'] == 'SOC_RECOVERED'])} times")
print(f"  PV output zero: {pv_zero_events} times")
print(f"  PV output active: {len(fault_df[fault_df['event'] == 'OUTPUT_ACTIVE'])} times")
print("\n" + "="*60)
print("✓ Files written: state_data_with_override.csv, sensor_data.csv, fault_log.csv")
print("="*60)

# ============================================================
#  FIGURES – DISPLAY ONLY (EXPORT CODE IS KEPT BUT NOT RUN)
# ============================================================

# ------------------------------------------------------------
#  FIGURE 1: Hardware‑only component states (no overrides)
# ------------------------------------------------------------
hw_components = [name + "_hw" for name in components.keys()]
fig1, ax1 = plt.subplots(figsize=(14, 6))
offset = 0
for name in hw_components:
    ax1.step(full_df.index, full_df[name] + offset, where="post",
             label=name.replace("_hw",""), linewidth=1.5)
    offset += 1.2
ax1.set_yticks([])
ax1.set_ylabel("Component State")
ax1.set_title("Hardware‑Only Component States (used for energy dispatch)")
ax1.legend(loc="upper right", ncol=2, fontsize=8)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
if EXPORT_FIGURES:
    plt.savefig("figure1_hardware_states.png", dpi=150)
plt.show()

# ------------------------------------------------------------
#  FIGURE 2: Two subplots – Overridden states (top) and Fault events (bottom)
# ------------------------------------------------------------
fig2, (ax2_top, ax2_bot) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# ---- Top: Component states with operational overrides ----
offset = 0
for name in components.keys():          # overridden columns
    ax2_top.step(full_df.index, full_df[name] + offset, where="post",
                 label=name, linewidth=1.5)
    offset += 1.2
ax2_top.set_yticks([])
ax2_top.set_ylabel("State")
ax2_top.set_title("Component States (WITH Operational Overrides)")
ax2_top.legend(loc="upper right", ncol=2, fontsize=8)
ax2_top.grid(True, alpha=0.3)

# ---- Bottom: Fault‑tree events, scaled for readability ----
ax2_bot.step(full_df.index, full_df["immediate_failure"] * 0.4, where="post",
             color="red", linewidth=2, label="Immediate Failure (0.4x)", alpha=0.7)
ax2_bot.step(full_df.index, full_df["islanded_failure"] * 0.8, where="post",
             color="orange", linewidth=2, label="Islanded Failure (0.8x)", alpha=0.7)
ax2_bot.step(full_df.index, full_df["loss_of_supply"] * 1.0, where="post",
             color="darkred", linewidth=2, label="Actual Loss of Supply", alpha=0.7)
ax2_bot.set_ylim(0, 1.2)
ax2_bot.set_ylabel("Scaled Events")
ax2_bot.set_title("Fault‑Tree Events (based on overridden states)")
ax2_bot.legend(loc="upper right")
ax2_bot.grid(True, alpha=0.3)

plt.tight_layout()
if EXPORT_FIGURES:
    plt.savefig("figure2_overridden_and_faults.png", dpi=150)
plt.show()

# ------------------------------------------------------------
#  FIGURE 3: Energy, Solar, Battery (three subplots)
# ------------------------------------------------------------
fig3, (ax3a, ax3b, ax3c) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# ---- 3a: Energy dispatch (load demand behind) ----
ax3a.plot(full_df.index, full_df["load_demand_kw"], color="red", linewidth=2,
          linestyle="--", label="Load Demand", zorder=1)
ax3a.fill_between(full_df.index, 0, full_df["pv_to_load_kw"],
                  alpha=0.6, color="gold", label="PV to Load", step="post", zorder=2)
ax3a.fill_between(full_df.index, full_df["pv_to_load_kw"],
                  full_df["pv_to_load_kw"] + full_df["battery_to_load_kw"],
                  alpha=0.6, color="green", label="Battery to Load", step="post", zorder=2)
ax3a.fill_between(full_df.index,
                  full_df["pv_to_load_kw"] + full_df["battery_to_load_kw"],
                  full_df["pv_to_load_kw"] + full_df["battery_to_load_kw"] + full_df["grid_to_load_kw"],
                  alpha=0.6, color="blue", label="Grid to Load", step="post", zorder=2)
ax3a.set_ylabel("Power (kW)")
ax3a.set_title("Energy Dispatch – Sources Meeting Load")
ax3a.legend(loc="upper right")
ax3a.grid(True, alpha=0.3)

# ---- 3b: Solar generation and usage ----
ax3b.fill_between(full_df.index, 0, full_df["solar_generation_kw"],
                  alpha=0.3, color="orange", label="Total PV Generation", step="post")
ax3b.plot(full_df.index, full_df["pv_to_load_kw"], color="red",
         linewidth=1, label="PV to Load", alpha=0.8)
ax3b.plot(full_df.index, full_df["pv_to_battery_kw"], color="green",
         linewidth=1, label="PV to Battery (CHARGING)", alpha=0.8)
ax3b.set_ylabel("Power (kW)")
ax3b.set_title("Solar Generation and Usage")
ax3b.legend(loc="upper right")
ax3b.grid(True, alpha=0.3)

# ---- 3c: Battery State of Charge ----
ax3c.plot(full_df.index, full_df["soc"], color="green", linewidth=2, label="Battery SOC (%)")
ax3c.fill_between(full_df.index, 0, full_df["soc"], alpha=0.2, color="green")
ax3c.axhline(y=0.1, color='red', linestyle='--', linewidth=1, alpha=0.5,
            label='Operational Threshold (0.1%)')
charging_times = full_df[full_df["pv_to_battery_kw"] > 0].index
if len(charging_times) > 0:
    ax3c.scatter(charging_times, full_df.loc[charging_times, "soc"],
                color="darkgreen", s=15, label=f"Charging ({len(charging_times)} hrs)",
                alpha=0.7, marker="^")
discharging_times = full_df[full_df["battery_to_load_kw"] > 0].index
if len(discharging_times) > 0:
    ax3c.scatter(discharging_times, full_df.loc[discharging_times, "soc"],
                color="orange", s=15, label=f"Discharging ({len(discharging_times)} hrs)",
                alpha=0.7, marker="v")
ax3c.set_ylabel("SOC (%)")
ax3c.set_xlabel("Time")
ax3c.set_title("Battery State of Charge")
ax3c.set_ylim(-5, 105)
ax3c.legend(loc="upper right")
ax3c.grid(True, alpha=0.3)

plt.tight_layout()
if EXPORT_FIGURES:
    plt.savefig("figure3_energy_battery.png", dpi=150)
plt.show()

# ------------------------------------------------------------
#  FIGURE 4: Comparison – Overridden vs Hardware (PV & Battery)
# ------------------------------------------------------------
fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

offset_pv = 0.1          # smaller offset
hw_pv = full_df["PV_Array_hw"]
ov_pv = full_df["PV_Array"]
ax4a.step(full_df.index, hw_pv, where="post",
          color="red", linewidth=2, label="PV_Array (hardware only)", alpha=0.7, linestyle='--')
ax4a.step(full_df.index, ov_pv * (1 + offset_pv), where="post",
          color="orange", linewidth=2, label="PV_Array (with override)", alpha=0.7)
ax4a.set_yticks([0, 1, 1 + offset_pv])
ax4a.set_yticklabels(["0", "1 (hw)", f"1+{offset_pv} (ov)"])
ax4a.set_ylabel("State")
ax4a.set_title("PV_Array – Hardware vs Overridden")
ax4a.legend(loc="upper right")
ax4a.grid(True, alpha=0.3)

offset_bat = 0.1
hw_bat = full_df["Battery_Pack_hw"]
ov_bat = full_df["Battery_Pack"]
ax4b.step(full_df.index, hw_bat, where="post",
          color="darkgreen", linewidth=2, label="Battery_Pack (hardware only)", alpha=0.7, linestyle='--')
ax4b.step(full_df.index, ov_bat * (1 + offset_bat), where="post",
          color="green", linewidth=2, label="Battery_Pack (with override)", alpha=0.7)
ax4b.set_yticks([0, 1, 1 + offset_bat])
ax4b.set_yticklabels(["0", "1 (hw)", f"1+{offset_bat} (ov)"])
ax4b.set_xlabel("Time")
ax4b.set_ylabel("State")
ax4b.set_title("Battery_Pack – Hardware vs Overridden")
ax4b.legend(loc="upper right")
ax4b.grid(True, alpha=0.3)

plt.tight_layout()
if EXPORT_FIGURES:
    plt.savefig("figure4_comparison_PV_Battery.png", dpi=150)
plt.show()

print("\n" + "="*60)
print("ALL FIGURES DISPLAYED – EXPORT CODE IS PRESENT BUT NOT ACTIVE (EXPORT_FIGURES = False)")
print("="*60)