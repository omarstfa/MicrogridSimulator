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

# np.random.seed(7)

# ============================================================
# COMPONENT DEFINITIONS (INTENTIONALLY HIGH FAILURE RATES)
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

def loss_of_supply(components):
    BE1 = not components["Grid"]["up"]
    BE2 = not components["PCC_Breaker"]["up"]
    BE3 = not components["Islanding_Controller"]["up"]
    BE4 = not components["PV_Array"]["up"]
    BE5 = not components["PV_Inverter"]["up"]
    BE6 = not components["Battery_Pack"]["up"]
    BE7 = not components["BMS"]["up"]
    BE8 = not components["PCS"]["up"]

    immediate_failure = BE1 and (BE2 or BE3)
    islanded_failure = BE1 and ((BE4 or BE5) and (BE6 or BE7 or BE8))

    return immediate_failure or islanded_failure

# ============================================================
# DATA STORAGE
# ============================================================

fault_log = []
state_log = []

soc = 80.0

# ============================================================
# MAIN SIMULATION LOOP
# ============================================================

for step in range(N_STEPS):
    current_time = start_time + pd.Timedelta(hours=step * DT_HOURS)

    for name, comp in components.items():
        if comp["up"] and attempt_failure(comp):
            comp["up"] = False
            fault_log.append([current_time, name, "FAILURE"])

        elif not comp["up"] and attempt_repair(comp):
            comp["up"] = True
            fault_log.append([current_time, name, "REPAIR"])

    if loss_of_supply(components):
        soc -= 2.0
    else:
        soc -= 0.1
    soc = np.clip(soc, 0, 100)

    row = {
        "loss_of_supply": int(loss_of_supply(components)),
        "soc": soc
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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

offset = 0
for name in components.keys():
    ax1.step(
        state_df.index,
        state_df[name] + offset,
        where="post",
        label=name
    )
    offset += 1.2

ax1.set_yticks([])
ax1.set_ylabel("Component State")
ax1.set_title("Component States Over Time (1 = UP, 0 = DOWN)")
ax1.legend(loc="upper right", ncol=2)
ax1.grid(True, alpha=0.3)

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
ax2.set_xlabel("Time")
ax2.set_title("System State Over Time")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Simulation complete.")
print("Loss of supply events:", state_df["loss_of_supply"].sum())
print("Files written: sensor_and_state_data.csv, fault_log.csv")