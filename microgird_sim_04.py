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

def can_battery_charge(components, solar_generation):
    """Check if battery can charge based on conditions"""
    condition1 = solar_generation > 0
    condition2 = components["PV_Array"]["up"]
    condition3 = components["PV_Inverter"]["up"]
    condition4 = components["Islanding_Controller"]["up"]
    condition5 = components["BMS"]["up"]
    
    return condition1 and condition2 and condition3 and condition4 and condition5

def calculate_load_demand(hour):
    """Calculate load demand based on time of day"""
    # Daily pattern: higher during day, lower at night
    daily_factor = np.sin(2 * np.pi * (hour % 24) / 24 - np.pi/2) * 0.5 + 0.8
    
    # Base load with variability
    load = BASE_LOAD_KW * daily_factor + (PEAK_LOAD_KW - BASE_LOAD_KW) * np.random.random() * LOAD_VARIABILITY
    
    return max(1.0, load)  # Minimum 1 kW load

def check_loss_of_supply(components, solar_generation, battery_kwh, load_demand):
    """Check if loss of supply occurs based on generation availability and component states"""
    
    # Check component availability for each generation source
    pv_available = components["PV_Array"]["up"] and components["PV_Inverter"]["up"]
    battery_available = components["Battery_Pack"]["up"] and components["PCS"]["up"] and components["BMS"]["up"]
    grid_available = components["Grid"]["up"] and components["PCC_Breaker"]["up"] and components["Islanding_Controller"]["up"]
    
    # Check generation availability
    pv_generation = solar_generation if pv_available else 0
    battery_generation = min(battery_kwh / DT_HOURS, BATTERY_MAX_DISCHARGE_RATE_KW) if battery_available else 0
    
    # Priority-based supply check
    # 1) PV alone
    if pv_generation >= load_demand:
        return False  # No loss of supply
    
    # 2) PV + Battery
    if pv_available and battery_available:
        if pv_generation + battery_generation >= load_demand:
            return False  # No loss of supply
    
    # 3) Battery alone
    if battery_generation >= load_demand:
        return False  # No loss of supply
    
    # 4) Grid
    if grid_available:
        return False  # No loss of supply
    
    # 5) No source available -> Loss of supply
    return True

# ============================================================
# DATA STORAGE
# ============================================================

fault_log = []
state_log = []

soc = 50.0
battery_kwh = soc/100 * BATTERY_CAPACITY_KWH

# ============================================================
# MAIN SIMULATION LOOP
# ============================================================

for step in range(N_STEPS):
    current_time = start_time + pd.Timedelta(hours=step * DT_HOURS)
    
    solar_generation = solar_df.loc[current_time, 'pv_generation_kw']
    load_demand = calculate_load_demand(step)

    for name, comp in components.items():
        if comp["up"] and attempt_failure(comp):
            comp["up"] = False
            fault_log.append([current_time, name, "FAILURE"])

        elif not comp["up"] and attempt_repair(comp):
            comp["up"] = True
            fault_log.append([current_time, name, "REPAIR"])

    # Battery charging logic (independent of loss of supply)
    charge_conditions_met = can_battery_charge(components, solar_generation)
    
    if charge_conditions_met:
        available_charge_power = min(solar_generation, BATTERY_MAX_CHARGE_RATE_KW)
        energy_to_add = available_charge_power * DT_HOURS * BATTERY_EFFICIENCY
        battery_kwh = min(BATTERY_CAPACITY_KWH, battery_kwh + energy_to_add)
        soc = (battery_kwh / BATTERY_CAPACITY_KWH) * 100
    
    # Battery SOC only goes to 0 if battery pack is down
    if not components["Battery_Pack"]["up"]:
        battery_kwh = 0
        soc = 0
    
    # Check loss of supply based on generation and load
    loss_of_supply = check_loss_of_supply(components, solar_generation, battery_kwh, load_demand)

    row = {
        "loss_of_supply": int(loss_of_supply),
        "soc": soc,
        "battery_kwh": battery_kwh,
        "solar_generation_kw": solar_generation,
        "load_demand_kw": load_demand,
        "can_charge": int(charge_conditions_met)
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
    freq="H"
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

# ============================================================
# PLOT COMPONENT STATES
# ============================================================

fig, axes = plt.subplots(4, 1, figsize=(16, 16))

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
    color="red"
)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(["Normal", "Loss of Supply"])
ax2.set_ylabel("System State")
ax2.set_title("System State (Loss of Supply Events)")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

# Plot 3: Energy Balance
ax3 = axes[2]
ax3.plot(
    state_df.index,
    state_df["solar_generation_kw"],
    color="orange",
    linewidth=1,
    label="Solar Generation (kW)",
    alpha=0.7
)
ax3.fill_between(
    state_df.index,
    0,
    state_df["solar_generation_kw"],
    alpha=0.2,
    color="orange"
)
ax3.plot(
    state_df.index,
    state_df["load_demand_kw"],
    color="blue",
    linewidth=1.5,
    label="Load Demand (kW)"
)

# Highlight loss of supply periods
loss_times = state_df[state_df["loss_of_supply"] == 1].index
if len(loss_times) > 0:
    for loss_time in loss_times:
        ax3.axvspan(loss_time, loss_time + pd.Timedelta(hours=1), alpha=0.3, color='red')

ax3.set_ylabel("Power (kW)")
ax3.set_title("Energy Generation vs Load Demand (Red areas = Loss of Supply)")
ax3.legend(loc="upper right")
ax3.grid(True, alpha=0.3)

# Plot 4: Battery State
ax4 = axes[3]
ax4.plot(
    state_df.index,
    state_df["soc"],
    color="green",
    linewidth=2,
    label="Battery SOC (%)"
)
ax4.fill_between(
    state_df.index,
    0,
    state_df["soc"],
    alpha=0.2,
    color="green"
)

# Add battery charging indicators
charging_times = state_df[state_df["can_charge"] == 1].index
if len(charging_times) > 0:
    ax4.scatter(
        charging_times,
        state_df.loc[charging_times, "soc"],
        color="darkgreen",
        s=10,
        label="Charging",
        alpha=0.6
    )

ax4.set_xlabel("Time")
ax4.set_ylabel("SOC (%)")
ax4.set_title("Battery State of Charge")
ax4.set_ylim(0, 105)
ax4.legend(loc="upper right")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Simulation complete.")
print(f"Loss of supply events: {state_df['loss_of_supply'].sum()}")
print(f"Total simulation time: {SIM_DAYS} days")
print(f"Final battery SOC: {soc:.1f}% ({battery_kwh:.1f} kWh)")
print(f"\nEnergy statistics:")
print(f"  Total solar energy: {state_df['solar_generation_kw'].sum():.1f} kWh")
print(f"  Total load demand: {state_df['load_demand_kw'].sum():.1f} kWh")
print(f"  Charging opportunities: {state_df['can_charge'].sum()} hours")
print(f"\nFiles written: sensor_and_state_data.csv, fault_log.csv")