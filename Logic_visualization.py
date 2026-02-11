import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

# ------------------------------------------------------------
#  FIGURE 1: FAULT LOGIC WITH OPERATIONAL OVERRIDES
# ------------------------------------------------------------
fig1, ax1 = plt.subplots(1, 1, figsize=(14, 10))
ax1.set_xlim(0, 12)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title("Microgrid Fault Logic (with Operational Overrides)", fontsize=16, fontweight='bold')

# ----- Hardware component states (left column) -----
hw_y_start = 8.5
hw_names = ["Grid", "PCC_Breaker", "Islanding_Controller",
            "PV_Array", "PV_Inverter", "Battery_Pack", "BMS", "PCS"]
hw_pos = {}
for i, name in enumerate(hw_names):
    y = hw_y_start - i * 0.8
    hw_pos[name] = (1.5, y)
    ax1.text(1.5, y, name, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", edgecolor="black"))

# ----- Operational overrides (right of hardware) -----
ov_pos = {}
# PV_Array override
ax1.text(3.5, hw_y_start - 3 * 0.8, "PV output > 0.01 ?", ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="black"))
ov_pos["PV_Array"] = (3.5, hw_y_start - 3 * 0.8)
# Battery_Pack override
ax1.text(3.5, hw_y_start - 5 * 0.8, "SOC > 0.1 ?", ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="black"))
ov_pos["Battery_Pack"] = (3.5, hw_y_start - 5 * 0.8)

# ----- AND gates for overrides -----
# PV_Array
ax1.annotate("", xy=(2.2, hw_y_start - 3 * 0.8), xytext=(1.5, hw_y_start - 3 * 0.8),
             arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"))
ax1.annotate("", xy=(3.5, hw_y_start - 3 * 0.8), xytext=(2.8, hw_y_start - 3 * 0.8),
             arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"))
ax1.text(2.5, hw_y_start - 3 * 0.8 + 0.2, "AND", ha='center', fontsize=9)

# Battery_Pack
ax1.annotate("", xy=(2.2, hw_y_start - 5 * 0.8), xytext=(1.5, hw_y_start - 5 * 0.8),
             arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"))
ax1.annotate("", xy=(3.5, hw_y_start - 5 * 0.8), xytext=(2.8, hw_y_start - 5 * 0.8),
             arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"))
ax1.text(2.5, hw_y_start - 5 * 0.8 + 0.2, "AND", ha='center', fontsize=9)

# ----- Overridden component states (middle column) -----
ovrd_y_start = hw_y_start
ovrd_names = ["Grid", "PCC_Breaker", "Islanding_Controller",
              "PV_Array (override)", "PV_Inverter", "Battery_Pack (override)", "BMS", "PCS"]
ovrd_pos = {}
for i, name in enumerate(ovrd_names):
    y = ovrd_y_start - i * 0.8
    ovrd_pos[name] = (5.5, y)
    # skip arrows for now
    if i == 3 or i == 5:
        facecol = "lightgreen"
    else:
        facecol = "lightblue"
    ax1.text(5.5, y, name, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor=facecol, edgecolor="black"))

# ----- Arrows from hardware to overrides (direct for unchanged) -----
for i, name in enumerate(hw_names):
    if name not in ["PV_Array", "Battery_Pack"]:
        ax1.annotate("", xy=(ovrd_pos[ovrd_names[i]][0] - 0.2, ovrd_pos[ovrd_names[i]][1]),
                     xytext=(hw_pos[name][0] + 0.3, hw_pos[name][1]),
                     arrowprops=dict(arrowstyle="->", lw=1, color="gray"))

# ----- Arrows from AND gates to overridden PV and Battery -----
ax1.annotate("", xy=(ovrd_pos["PV_Array (override)"][0] - 0.2, ovrd_pos["PV_Array (override)"][1]),
             xytext=(ov_pos["PV_Array"][0] + 0.3, ov_pos["PV_Array"][1]),
             arrowprops=dict(arrowstyle="->", lw=1.5, color="blue"))
ax1.annotate("", xy=(ovrd_pos["Battery_Pack (override)"][0] - 0.2, ovrd_pos["Battery_Pack (override)"][1]),
             xytext=(ov_pos["Battery_Pack"][0] + 0.3, ov_pos["Battery_Pack"][1]),
             arrowprops=dict(arrowstyle="->", lw=1.5, color="blue"))

# ----- Fault Tree Events (right column) -----
ft_y_start = 7.0
# Immediate failure
ax1.text(9.0, ft_y_start, "Immediate Failure", ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="salmon", edgecolor="black"))
ax1.text(7.0, ft_y_start - 0.5, "Grid down AND (PCC_Breaker down OR\nIslanding_Controller down)",
         ha='center', va='center', fontsize=9, style='italic')
# Islanded failure
ax1.text(9.0, ft_y_start - 2.0, "Islanded Failure", ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="salmon", edgecolor="black"))
ax1.text(7.0, ft_y_start - 2.7, "Grid down AND (PV system down) AND (Battery system down)",
         ha='center', va='center', fontsize=9, style='italic')

# ----- Arrows from overridden states to fault events -----
# Immediate failure: Grid, PCC_Breaker, Islanding_Controller
ax1.annotate("", xy=(8.0, ft_y_start), xytext=(ovrd_pos["Grid"][0] + 0.3, ovrd_pos["Grid"][1]),
             arrowprops=dict(arrowstyle="->", lw=1, color="red"))
ax1.annotate("", xy=(8.0, ft_y_start - 0.1), xytext=(ovrd_pos["PCC_Breaker"][0] + 0.3, ovrd_pos["PCC_Breaker"][1]),
             arrowprops=dict(arrowstyle="->", lw=1, color="red"))
ax1.annotate("", xy=(8.0, ft_y_start - 0.2), xytext=(ovrd_pos["Islanding_Controller"][0] + 0.3, ovrd_pos["Islanding_Controller"][1]),
             arrowprops=dict(arrowstyle="->", lw=1, color="red"))

# Islanded failure: Grid, PV system, Battery system
ax1.annotate("", xy=(8.0, ft_y_start - 2.0), xytext=(ovrd_pos["Grid"][0] + 0.3, ovrd_pos["Grid"][1]),
             arrowprops=dict(arrowstyle="->", lw=1, color="darkorange"))
ax1.annotate("", xy=(8.0, ft_y_start - 2.1), xytext=(ovrd_pos["PV_Array (override)"][0] + 0.3, ovrd_pos["PV_Array (override)"][1]),
             arrowprops=dict(arrowstyle="->", lw=1, color="darkorange"))
ax1.annotate("", xy=(8.0, ft_y_start - 2.2), xytext=(ovrd_pos["PV_Inverter"][0] + 0.3, ovrd_pos["PV_Inverter"][1]),
             arrowprops=dict(arrowstyle="->", lw=1, color="darkorange"))
ax1.annotate("", xy=(8.0, ft_y_start - 2.3), xytext=(ovrd_pos["Battery_Pack (override)"][0] + 0.3, ovrd_pos["Battery_Pack (override)"][1]),
             arrowprops=dict(arrowstyle="->", lw=1, color="darkorange"))
ax1.annotate("", xy=(8.0, ft_y_start - 2.4), xytext=(ovrd_pos["BMS"][0] + 0.3, ovrd_pos["BMS"][1]),
             arrowprops=dict(arrowstyle="->", lw=1, color="darkorange"))
ax1.annotate("", xy=(8.0, ft_y_start - 2.5), xytext=(ovrd_pos["PCS"][0] + 0.3, ovrd_pos["PCS"][1]),
             arrowprops=dict(arrowstyle="->", lw=1, color="darkorange"))

# ----- Title and notes -----
ax1.text(6, 1, "Operational overrides are applied to PV_Array and Battery_Pack\n"
              "based on real‑time PV output and battery SOC.\n"
              "Fault tree events are computed from these *overridden* states.",
         ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="ivory"))

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
#  FIGURE 2: ENERGY DISPATCH LOGIC (PRIORITY FLOW)
# ------------------------------------------------------------
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 12)
ax2.axis('off')
ax2.set_title("Microgrid Energy Dispatch Logic", fontsize=16, fontweight='bold')

# ----- Sources (top) -----
ax2.text(2, 11, "PV Generation", ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="gold", edgecolor="black"))
ax2.text(5, 11, "Battery Storage", ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", edgecolor="black"))
ax2.text(8, 11, "Grid", ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", edgecolor="black"))

# ----- Load (bottom) -----
ax2.text(5, 2, "Load Demand", ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", edgecolor="black"))

# ----- Decision boxes -----
# Priority 1: PV to Load
ax2.text(2, 9, "PV available &\nPV > 0.01 ?", ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="black"))
ax2.annotate("", xy=(2, 10.3), xytext=(2, 10.7), arrowprops=dict(arrowstyle="->", lw=1.5))
ax2.annotate("", xy=(2, 8.3), xytext=(2, 8.7), arrowprops=dict(arrowstyle="->", lw=1.5))
ax2.text(3.5, 9, "yes → use PV to meet load", fontsize=9, color="green")
ax2.text(3.5, 8.5, "no → skip", fontsize=9, color="gray")

# Excess PV
ax2.text(2, 7, "Excess PV after\nmeeting load ?", ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="black"))
ax2.annotate("", xy=(2, 8.3), xytext=(2, 8.7), arrowprops=dict(arrowstyle="->", lw=1))
ax2.text(3.5, 7, "yes → can charge battery", fontsize=9, color="green")
ax2.text(3.5, 6.5, "no → skip", fontsize=9, color="gray")

# Battery charge conditions
ax2.text(5, 6, "Battery hardware OK &\nSOC < 100% &\nIslanding_Controller up ?",
         ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="black"))
ax2.annotate("", xy=(2.5, 7), xytext=(2, 7), arrowprops=dict(arrowstyle="->", lw=1, color="green"))
ax2.text(6.5, 6, "yes → charge battery\n(rate limited, efficiency applied)", fontsize=9, color="green")
ax2.text(6.5, 5.5, "no → no charge", fontsize=9, color="gray")

# Priority 2: Battery to Load
ax2.text(5, 4, "Load still unmet ?\nBattery can discharge ?", ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="black"))
ax2.annotate("", xy=(5, 5.3), xytext=(5, 5.7), arrowprops=dict(arrowstyle="->", lw=1))
ax2.text(6.5, 4, "yes → discharge battery\n(rate limited, efficiency applied)", fontsize=9, color="green")
ax2.text(6.5, 3.5, "no → skip", fontsize=9, color="gray")

# Priority 3: Grid to Load
ax2.text(8, 3, "Load still unmet ?\nGrid available ?", ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="black"))
ax2.annotate("", xy=(5.5, 4), xytext=(5, 4), arrowprops=dict(arrowstyle="->", lw=1))
ax2.text(9, 3, "yes → grid meets remaining load", fontsize=9, color="green")
ax2.text(9, 2.5, "no → loss of supply", fontsize=9, color="red")

# Arrows to load
ax2.annotate("", xy=(5, 2.5), xytext=(2, 8.3), arrowprops=dict(arrowstyle="->", lw=1, color="gold"))
ax2.annotate("", xy=(5, 2.5), xytext=(5, 4.5), arrowprops=dict(arrowstyle="->", lw=1, color="green"))
ax2.annotate("", xy=(5, 2.5), xytext=(8, 3.5), arrowprops=dict(arrowstyle="->", lw=1, color="blue"))

# ----- Additional notes -----
ax2.text(5, 0.5, "All power flows respect hardware states and operational limits.\n"
                "PV to battery only if Islanding_Controller is up.\n"
                "Grid is used only as last resort when load cannot be met otherwise.",
         ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="ivory"))

plt.tight_layout()
plt.show()