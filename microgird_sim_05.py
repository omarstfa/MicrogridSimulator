# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 00:10:30 2026

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# SIMULATION PARAMETERS
# ============================================================

SIM_DAYS = 180
DT_HOURS = 1.0
N_STEPS = int((SIM_DAYS * 24) / DT_HOURS)

start_time = pd.Timestamp("2026-01-01 00:00:00")

np.random.seed(7)

# Load solar generation data
try:
    solar_df = pd.read_csv('solar_generation.csv', index_col='timestamp', parse_dates=True)
    print("Loaded solar generation data")
except FileNotFoundError:
    print("ERROR: solar_generation.csv not found. Please run solar_generation.py first.")
    exit()

# System parameters
BATTERY_CAPACITY_KWH = 20
BATTERY_MAX_CHARGE_RATE_KW = 5
BATTERY_MAX_DISCHARGE_RATE_KW = 5
BATTERY_EFFICIENCY = 0.95
PV_CAPACITY_KW = 10

# Load demand parameters (residential microgrid)
BASE_LOAD_KW = 2.5  # Base load
PEAK_LOAD_KW = 5    # Peak load
LOAD_VARIABILITY = 0.3  # Load variability factor

# ============================================================
# COMPONENT DEFINITIONS
# ============================================================

components = {
    "Grid": {
        "lambda": 0.08 / 24,
        "mu": 0.4 / 24,
        "up": True
    },
    "PCC_Breaker": {
        "lambda": 5e-3,
        "mu": 0.3,
        "up": True
    },
    "Islanding_Controller": {
        "lambda": 7e-3,
        "mu": 0.25,
        "up": True
    },
    "PV_Array": {
        "lambda": 5e-3,
        "mu": 0.05,
        "up": True
    },
    "PV_Inverter": {
        "lambda": 8e-3,
        "mu": 0.15,
        "up": True
    },
    "Battery_Pack": {
        "lambda": 6e-3,
        "mu": 0.05,
        "up": True
    },
    "BMS": {
        "lambda": 7e-3,
        "mu": 0.25,
        "up": True
    },
    "PCS": {
        "lambda": 7e-3,
        "mu": 0.25,
        "up": True
    }
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def attempt_failure(comp):
    return np.random.rand() < comp["lambda"] * DT_HOURS

def attempt_repair(comp):
    return np.random.rand() < comp["mu"] * DT_HOURS

def calculate_load_demand(hour):
    """Calculate load demand based on time of day"""
    # Daily pattern: higher during day, lower at night
    daily_factor = np.sin(2 * np.pi * (hour % 24) / 24 - np.pi/2) * 0.5 + 0.8
    
    # Base load with variability
    load = BASE_LOAD_KW * daily_factor + (PEAK_LOAD_KW - BASE_LOAD_KW) * np.random.random() * LOAD_VARIABILITY
    
    return max(1.0, load)  # Minimum 1 kW load

# ============================================================
# DATA STORAGE
# ============================================================

fault_log = []
state_log = []

soc = 50.0
battery_kwh = soc/100 * BATTERY_CAPACITY_KWH

# Track previous states for failure detection
prev_soc = soc
prev_pv_output = 0

# ============================================================
# MAIN SIMULATION LOOP
# ============================================================

for step in range(N_STEPS):
    current_time = start_time + pd.Timedelta(hours=step * DT_HOURS)
    
    solar_generation = solar_df.loc[current_time, 'pv_generation_kw']
    load_demand = calculate_load_demand(step)

    # ========================================================
    # COMPONENT FAILURE AND REPAIR LOGIC
    # ========================================================
    for name, comp in components.items():
        if comp["up"] and attempt_failure(comp):
            comp["up"] = False
            fault_log.append([current_time, name, "FAILURE"])

        elif not comp["up"] and attempt_repair(comp):
            comp["up"] = True
            fault_log.append([current_time, name, "REPAIR"])

    # ========================================================
    # ENERGY DISPATCH LOGIC (REALISTIC)
    # ========================================================
    
    # Check component availability for each source
    pv_available = components["PV_Array"]["up"] and components["PV_Inverter"]["up"]
    battery_available = components["Battery_Pack"]["up"] and components["PCS"]["up"] and components["BMS"]["up"]
    grid_available = components["Grid"]["up"] and components["PCC_Breaker"]["up"] and components["Islanding_Controller"]["up"]
    
    # Available power from each source
    pv_power = solar_generation if pv_available else 0
    battery_max_discharge = min(battery_kwh / DT_HOURS, BATTERY_MAX_DISCHARGE_RATE_KW) if battery_available else 0
    
    # Initialize energy flows
    pv_to_load = 0
    battery_to_load = 0
    grid_to_load = 0
    pv_to_battery = 0
    
    remaining_load = load_demand
    
    # Priority 1: Use PV to meet load first
    if pv_power > 0:
        pv_to_load = min(pv_power, remaining_load)
        remaining_load -= pv_to_load
        pv_excess = pv_power - pv_to_load
    else:
        pv_excess = 0
    
    # Priority 2: Charge battery with excess PV (if available and battery can charge)
    can_charge = (pv_excess > 0 and 
                  battery_available and 
                  components["Islanding_Controller"]["up"] and 
                  components["BMS"]["up"] and
                  battery_kwh < BATTERY_CAPACITY_KWH)
    
    if can_charge:
        charge_power = min(pv_excess, BATTERY_MAX_CHARGE_RATE_KW)
        energy_to_add = charge_power * DT_HOURS * BATTERY_EFFICIENCY
        space_available = BATTERY_CAPACITY_KWH - battery_kwh
        energy_to_add = min(energy_to_add, space_available)
        pv_to_battery = energy_to_add / (DT_HOURS * BATTERY_EFFICIENCY)  # Power used for charging
        battery_kwh += energy_to_add
    
    # Priority 3: Use battery to supplement if load not fully met by PV
    if remaining_load > 0.01 and battery_available and battery_kwh > 0:
        battery_to_load = min(battery_max_discharge, remaining_load)
        energy_discharged = battery_to_load * DT_HOURS / BATTERY_EFFICIENCY
        energy_discharged = min(energy_discharged, battery_kwh)  # Don't overdraw
        battery_to_load = energy_discharged * BATTERY_EFFICIENCY / DT_HOURS  # Actual power delivered
        battery_kwh -= energy_discharged
        remaining_load -= battery_to_load
    
    # Priority 4: Use grid as backup if load still not met
    if remaining_load > 0.01 and grid_available:
        grid_to_load = remaining_load
        remaining_load = 0
    
    # Priority 5: Loss of supply if load cannot be met
    loss_of_supply = remaining_load > 0.01  # Small tolerance for floating point
    
    # Update SOC
    soc = (battery_kwh / BATTERY_CAPACITY_KWH) * 100
    
    # ========================================================
    # FAILURE LOGGING FOR ZERO STATES
    # ========================================================
    
    # Log battery SOC reaching 0 (transition from non-zero to zero)
    if prev_soc > 0.1 and soc <= 0.1:
        fault_log.append([current_time, "Battery_Pack", "SOC_DEPLETED"])
    
    # Log battery SOC recovering from 0 (transition from zero to non-zero)
    if prev_soc <= 0.1 and soc > 0.1:
        fault_log.append([current_time, "Battery_Pack", "SOC_RECOVERED"])
    
    # Log PV output reaching 0 (transition from non-zero to zero)
    if prev_pv_output > 0.01 and solar_generation <= 0.01:
        fault_log.append([current_time, "PV_Array", "OUTPUT_ZERO"])
    
    # Log PV output recovering from 0 (transition from zero to non-zero)
    if prev_pv_output <= 0.01 and solar_generation > 0.01:
        fault_log.append([current_time, "PV_Array", "OUTPUT_ACTIVE"])
    
    # Update previous states for next iteration
    prev_soc = soc
    prev_pv_output = solar_generation
    
    # ========================================================
    # LOG STATE DATA
    # ========================================================
    
    row = {
        "loss_of_supply": int(loss_of_supply),
        "soc": soc,
        "battery_kwh": battery_kwh,
        "solar_generation_kw": solar_generation,
        "load_demand_kw": load_demand,
        "pv_to_load_kw": pv_to_load,
        "pv_to_battery_kw": pv_to_battery,
        "battery_to_load_kw": battery_to_load,
        "grid_to_load_kw": grid_to_load,
        "unmet_load_kw": remaining_load,
        "can_charge": int(can_charge)
    }

    for name, comp in components.items():
        row[name] = int(comp["up"])

    state_log.append(row)

# ============================================================
# CREATE DATAFRAMES
# ============================================================

state_df = pd.DataFrame(state_log)
state_df.index = pd.date_range(
    start=start_time,
    periods=len(state_df),
    freq="h"
)

fault_df = pd.DataFrame(
    fault_log,
    columns=["time", "component", "event"]
).set_index("time")

# ============================================================
# EXPORT
# ============================================================

state_df.to_csv("sensor_and_state_data.csv")
fault_df.to_csv("fault_log.csv")

print("\n" + "="*60)
print("EXPORT COMPLETE")
print("="*60)
print(f"✓ sensor_and_state_data.csv ({len(state_df)} rows)")
print(f"✓ fault_log.csv ({len(fault_df)} events)")

# ============================================================
# PLOT COMPONENT STATES
# ============================================================

fig, axes = plt.subplots(5, 1, figsize=(16, 18))

# Plot 1: Component States
ax1 = axes[0]
offset = 0
for name in components.keys():
    ax1.step(
        state_df.index,
        state_df[name] + offset,
        where="post",
        label=name,
        linewidth=1.5
    )
    offset += 1.2

ax1.set_yticks([])
ax1.set_ylabel("Component State")
ax1.set_title("Component States Over Time (1 = UP, 0 = DOWN)")
ax1.legend(loc="upper right", ncol=2, fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: System State
ax2 = axes[1]
ax2.step(
    state_df.index,
    state_df["loss_of_supply"],
    where="post",
    color="red",
    linewidth=2,
    label="Loss of Supply"
)
ax2.fill_between(
    state_df.index,
    0,
    state_df["loss_of_supply"],
    alpha=0.3,
    color="red",
    step="post"
)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(["Normal", "Loss of Supply"])
ax2.set_ylabel("System State")
ax2.set_title("System State (Loss of Supply Events)")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

# Plot 3: Energy Flows
ax3 = axes[2]
ax3.fill_between(
    state_df.index,
    0,
    state_df["pv_to_load_kw"],
    alpha=0.6,
    color="gold",
    label="PV to Load",
    step="post"
)
ax3.fill_between(
    state_df.index,
    state_df["pv_to_load_kw"],
    state_df["pv_to_load_kw"] + state_df["battery_to_load_kw"],
    alpha=0.6,
    color="green",
    label="Battery to Load",
    step="post"
)
ax3.fill_between(
    state_df.index,
    state_df["pv_to_load_kw"] + state_df["battery_to_load_kw"],
    state_df["pv_to_load_kw"] + state_df["battery_to_load_kw"] + state_df["grid_to_load_kw"],
    alpha=0.6,
    color="blue",
    label="Grid to Load",
    step="post"
)
ax3.plot(
    state_df.index,
    state_df["load_demand_kw"],
    color="red",
    linewidth=2,
    label="Load Demand",
    linestyle="--"
)
ax3.set_ylabel("Power (kW)")
ax3.set_title("Energy Dispatch - Sources Meeting Load")
ax3.legend(loc="upper right")
ax3.grid(True, alpha=0.3)

# Plot 4: PV Generation vs Usage
ax4 = axes[3]
ax4.fill_between(
    state_df.index,
    0,
    state_df["solar_generation_kw"],
    alpha=0.4,
    color="orange",
    label="Total PV Generation",
    step="post"
)
ax4.plot(
    state_df.index,
    state_df["pv_to_load_kw"],
    color="red",
    linewidth=1.5,
    label="PV to Load"
)
ax4.plot(
    state_df.index,
    state_df["pv_to_battery_kw"],
    color="green",
    linewidth=1.5,
    label="PV to Battery"
)

ax4.set_ylabel("Power (kW)")
ax4.set_title("Solar Generation and Usage")
ax4.legend(loc="upper right")
ax4.grid(True, alpha=0.3)

# Plot 5: Battery State
ax5 = axes[4]
ax5.plot(
    state_df.index,
    state_df["soc"],
    color="green",
    linewidth=2,
    label="Battery SOC (%)"
)
ax5.fill_between(
    state_df.index,
    0,
    state_df["soc"],
    alpha=0.2,
    color="green"
)

# Add battery charging indicators
charging_times = state_df[state_df["can_charge"] == 1].index
if len(charging_times) > 0:
    ax5.scatter(
        charging_times,
        state_df.loc[charging_times, "soc"],
        color="darkgreen",
        s=10,
        label="Charging",
        alpha=0.6
    )

# Add battery discharging indicators
discharging_times = state_df[state_df["battery_to_load_kw"] > 0].index
if len(discharging_times) > 0:
    ax5.scatter(
        discharging_times,
        state_df.loc[discharging_times, "soc"],
        color="orange",
        s=10,
        label="Discharging",
        alpha=0.6
    )

ax5.set_xlabel("Time")
ax5.set_ylabel("SOC (%)")
ax5.set_title("Battery State of Charge")
ax5.set_ylim(0, 105)
ax5.legend(loc="upper right")
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("microgrid_simulation_results.png", dpi=150, bbox_inches='tight')
print("\n✓ microgrid_simulation_results.png saved")
plt.show()

# ============================================================
# SUMMARY STATISTICS
# ============================================================

print("\n" + "="*60)
print("SIMULATION SUMMARY")
print("="*60)
print(f"Simulation period: {SIM_DAYS} days ({N_STEPS} hours)")
print(f"\nLOSS OF SUPPLY:")
print(f"  Total events: {state_df['loss_of_supply'].sum()} hours")
print(f"  Percentage: {(state_df['loss_of_supply'].sum() / N_STEPS * 100):.2f}%")

print(f"\nENERGY STATISTICS:")
print(f"  Total solar generation: {state_df['solar_generation_kw'].sum():.1f} kWh")
print(f"  Total load demand: {state_df['load_demand_kw'].sum():.1f} kWh")
print(f"  Total unmet load: {state_df['unmet_load_kw'].sum():.1f} kWh")

print(f"\nENERGY DISPATCH:")
print(f"  PV to load: {state_df['pv_to_load_kw'].sum():.1f} kWh ({state_df['pv_to_load_kw'].sum() / state_df['load_demand_kw'].sum() * 100:.1f}%)")
print(f"  PV to battery: {state_df['pv_to_battery_kw'].sum():.1f} kWh")
print(f"  Battery to load: {state_df['battery_to_load_kw'].sum():.1f} kWh ({state_df['battery_to_load_kw'].sum() / state_df['load_demand_kw'].sum() * 100:.1f}%)")
print(f"  Grid to load: {state_df['grid_to_load_kw'].sum():.1f} kWh ({state_df['grid_to_load_kw'].sum() / state_df['load_demand_kw'].sum() * 100:.1f}%)")

print(f"\nBATTERY STATISTICS:")
print(f"  Final SOC: {soc:.1f}% ({battery_kwh:.1f} kWh)")
print(f"  Charging hours: {state_df['can_charge'].sum()}")
print(f"  Discharging hours: {(state_df['battery_to_load_kw'] > 0).sum()}")
print(f"  Average SOC: {state_df['soc'].mean():.1f}%")
print(f"  Min SOC: {state_df['soc'].min():.1f}%")
print(f"  Max SOC: {state_df['soc'].max():.1f}%")

print(f"\nCOMPONENT FAILURES:")
failure_counts = fault_df[fault_df['event'] == 'FAILURE'].groupby('component').size()
repair_counts = fault_df[fault_df['event'] == 'REPAIR'].groupby('component').size()
soc_depleted = len(fault_df[fault_df['event'] == 'SOC_DEPLETED'])
pv_zero_events = len(fault_df[fault_df['event'] == 'OUTPUT_ZERO'])

for comp in components.keys():
    failures = failure_counts.get(comp, 0)
    repairs = repair_counts.get(comp, 0)
    print(f"  {comp}: {failures} failures, {repairs} repairs")

print(f"\nSPECIAL EVENTS:")
print(f"  Battery SOC depleted: {soc_depleted} times")
print(f"  Battery SOC recovered: {len(fault_df[fault_df['event'] == 'SOC_RECOVERED'])} times")
print(f"  PV output zero: {pv_zero_events} times")
print(f"  PV output active: {len(fault_df[fault_df['event'] == 'OUTPUT_ACTIVE'])} times")

print("\n" + "="*60)
print("Files written:")
print("  • sensor_and_state_data.csv")
print("  • fault_log.csv")
print("  • microgrid_simulation_results.png")
print("="*60)