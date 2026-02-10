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
BASE_LOAD_KW = 3  # Base load
PEAK_LOAD_KW = 8   # Peak load
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

def calculate_fault_tree_events(comp_states):
    """
    Calculate intermediate fault tree events based on component states
    
    Fault Tree Logic:
    immediate_failure = Grid down AND (PCC_Breaker OR Islanding_Controller down)
    islanded_failure = Grid down AND (PV system down) AND (Battery system down)
    loss_of_supply = immediate_failure OR islanded_failure
    """
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
    
    return {
        "immediate_failure": immediate_failure,
        "islanded_failure": islanded_failure
    }

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
    # COMPONENT FAILURE AND REPAIR LOGIC (RANDOM FAILURES)
    # ========================================================
    for name, comp in components.items():
        # Skip PV_Array and Battery_Pack - their states will be overridden by operational conditions
        if name in ["PV_Array", "Battery_Pack"]:
            continue
            
        if comp["up"] and attempt_failure(comp):
            comp["up"] = False
            fault_log.append([current_time, name, "FAILURE"])

        elif not comp["up"] and attempt_repair(comp):
            comp["up"] = True
            fault_log.append([current_time, name, "REPAIR"])

    # ========================================================
    # OPERATIONAL STATE OVERRIDES
    # ========================================================
    
    # PV_Array state: DOWN when solar output is 0
    pv_array_operational = solar_generation > 0.01
    
    # Track PV_Array state changes due to operational conditions
    if components["PV_Array"]["up"] and not pv_array_operational:
        components["PV_Array"]["up"] = False
        fault_log.append([current_time, "PV_Array", "OUTPUT_ZERO"])
    elif not components["PV_Array"]["up"] and pv_array_operational:
        # Only restore if it wasn't failed due to random failure
        # Check if there was a random failure that hasn't been repaired
        components["PV_Array"]["up"] = True
        fault_log.append([current_time, "PV_Array", "OUTPUT_ACTIVE"])
    
    # Battery_Pack will be set later based on SOC after energy dispatch

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
    # BATTERY_PACK STATE OVERRIDE BASED ON SOC
    # ========================================================
    
    battery_pack_operational = soc > 0.1  # Battery considered operational if SOC > 0.1%
    
    # Track Battery_Pack state changes due to SOC depletion
    if components["Battery_Pack"]["up"] and not battery_pack_operational:
        components["Battery_Pack"]["up"] = False
        fault_log.append([current_time, "Battery_Pack", "SOC_DEPLETED"])
    elif not components["Battery_Pack"]["up"] and battery_pack_operational:
        components["Battery_Pack"]["up"] = True
        fault_log.append([current_time, "Battery_Pack", "SOC_RECOVERED"])
    
    # ========================================================
    # CALCULATE FAULT TREE INTERMEDIATE EVENTS
    # ========================================================
    
    component_states = {name: comp["up"] for name, comp in components.items()}
    fault_tree_events = calculate_fault_tree_events(component_states)
    
    # ========================================================
    # LOG STATE DATA
    # ========================================================
    
    row = {
        "loss_of_supply": int(loss_of_supply),
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

fig, axes = plt.subplots(6, 1, figsize=(16, 20))

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

# Plot 2: Fault Tree Intermediate Events
ax2 = axes[1]
ax2.fill_between(
    state_df.index,
    0,
    state_df["immediate_failure"],
    alpha=0.5,
    color="red",
    label="Immediate Failure",
    step="post"
)
ax2.fill_between(
    state_df.index,
    state_df["immediate_failure"],
    state_df["immediate_failure"] + state_df["islanded_failure"],
    alpha=0.5,
    color="orange",
    label="Islanded Failure",
    step="post"
)
ax2.step(
    state_df.index,
    state_df["loss_of_supply"],
    where="post",
    color="darkred",
    linewidth=2,
    label="Loss of Supply",
    linestyle="--"
)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(["Normal", "Failure"])
ax2.set_ylabel("Failure Mode")
ax2.set_title("Fault Tree Intermediate Events")
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

# Overlay PV_Array state
pv_down_times = state_df[state_df["PV_Array"] == 0].index
if len(pv_down_times) > 0:
    for pv_time in pv_down_times:
        ax4.axvspan(pv_time, pv_time + pd.Timedelta(hours=1), alpha=0.2, color='gray')

ax4.set_ylabel("Power (kW)")
ax4.set_title("Solar Generation and Usage (Gray = PV Array Down)")
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

# Overlay Battery_Pack state (when down due to SOC = 0)
battery_down_times = state_df[state_df["Battery_Pack"] == 0].index
if len(battery_down_times) > 0:
    for bat_time in battery_down_times:
        ax5.axvspan(bat_time, bat_time + pd.Timedelta(hours=1), alpha=0.3, color='red')

ax5.set_ylabel("SOC (%)")
ax5.set_title("Battery State of Charge (Red areas = Battery Pack Down)")
ax5.set_ylim(0, 105)
ax5.legend(loc="upper right")
ax5.grid(True, alpha=0.3)

# Plot 6: Component Availability Breakdown
ax6 = axes[5]
# Calculate availability for key components
time_hours = len(state_df)
component_availability = {}
for comp_name in components.keys():
    uptime = state_df[comp_name].sum()
    availability = (uptime / time_hours) * 100
    component_availability[comp_name] = availability

# Create bar chart
comp_names = list(component_availability.keys())
availabilities = list(component_availability.values())
colors = ['green' if av > 95 else 'orange' if av > 90 else 'red' for av in availabilities]

bars = ax6.barh(comp_names, availabilities, color=colors, alpha=0.7)
ax6.set_xlabel("Availability (%)")
ax6.set_title("Component Availability Over Simulation Period")
ax6.set_xlim(0, 100)
ax6.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (name, av) in enumerate(zip(comp_names, availabilities)):
    ax6.text(av + 1, i, f'{av:.1f}%', va='center', fontsize=9)

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

print(f"\nFAULT TREE ANALYSIS:")
print(f"  Loss of Supply: {state_df['loss_of_supply'].sum()} hours ({(state_df['loss_of_supply'].sum() / N_STEPS * 100):.2f}%)")
print(f"  Immediate Failure: {state_df['immediate_failure'].sum()} hours ({(state_df['immediate_failure'].sum() / N_STEPS * 100):.2f}%)")
print(f"  Islanded Failure: {state_df['islanded_failure'].sum()} hours ({(state_df['islanded_failure'].sum() / N_STEPS * 100):.2f}%)")

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
print(f"  Hours at SOC = 0: {(state_df['soc'] < 0.1).sum()}")

print(f"\nCOMPONENT AVAILABILITY:")
for comp_name in components.keys():
    uptime = state_df[comp_name].sum()
    availability = (uptime / time_hours) * 100
    print(f"  {comp_name}: {availability:.2f}%")

print(f"\nCOMPONENT EVENTS:")
failure_counts = fault_df[fault_df['event'] == 'FAILURE'].groupby('component').size()
repair_counts = fault_df[fault_df['event'] == 'REPAIR'].groupby('component').size()
soc_depleted = len(fault_df[fault_df['event'] == 'SOC_DEPLETED'])
pv_zero_events = len(fault_df[fault_df['event'] == 'OUTPUT_ZERO'])

for comp in components.keys():
    failures = failure_counts.get(comp, 0)
    repairs = repair_counts.get(comp, 0)
    if failures > 0 or repairs > 0:
        print(f"  {comp}: {failures} failures, {repairs} repairs")

print(f"\nOPERATIONAL STATE EVENTS:")
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