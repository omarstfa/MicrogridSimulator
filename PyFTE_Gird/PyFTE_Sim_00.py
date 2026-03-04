"""
PyFTE_sim_00.py

Simulate the reliability of a microgrid using only:
  - fault_tree_expression.csv   (Boolean expression for loss_of_supply)
  - distribution_parameters.csv (exponential failure/repair rates for each basic event)

Produces:
  - SAIFI, SAIDI, CAIDI, ENS, AENS
  - Optional plot of loss‑of‑supply events
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Load input data
# ----------------------------------------------------------------------
try:
    ft_df = pd.read_csv('fault_tree_expression_10.csv')
    dist_df = pd.read_csv('distribution_parameters_10.csv')
except FileNotFoundError as e:
    print("Error: missing CSV file(s). Make sure both")
    print("  fault_tree_expression.csv  and  distribution_parameters.csv")
    print("are in the current directory.")
    raise e

# Top event name (for reference only)
top_event_name = ft_df['Top_Event'].iloc[0]
fault_expr = ft_df['Factored_Expression'].iloc[0]

# Convert the expression to Python syntax: '·' → ' and ', '+' → ' or '
py_expr = fault_expr.replace('·', ' and ').replace('+', ' or ')

# ----------------------------------------------------------------------
# 2. Build dictionaries for failure rate (λ) and repair rate (μ)
#    from the exponential parameters in distribution_parameters.csv
# ----------------------------------------------------------------------
lambda_dict = {}
mu_dict = {}

for _, row in dist_df.iterrows():
    be = row['ID']                     # e.g., 'BE1'
    lambda_dict[be] = row['Exp_λ_per_h']
    mttr = row['MTTR_h']
    # repair rate = 1 / MTTR (if MTTR > 0 and finite)
    if mttr > 0 and not np.isinf(mttr):
        mu_dict[be] = 1.0 / mttr
    else:
        mu_dict[be] = 0.0              # no repair possible

# ----------------------------------------------------------------------
# 3. Simulation parameters
# ----------------------------------------------------------------------
SIM_YEARS = 5
DT_HOURS = 1.0
TOTAL_HOURS = int(SIM_YEARS * 365 * 24 / DT_HOURS)

np.random.seed(42)                     # for reproducibility

# Initial state: all components are up (working) → failed = False
states = {be: False for be in lambda_dict.keys()}

# Array to record the top event at each hour
top_event = np.zeros(TOTAL_HOURS, dtype=int)

# ----------------------------------------------------------------------
# 4. Main simulation loop
# ----------------------------------------------------------------------
print("Simulating {} years ({} hours)...".format(SIM_YEARS, TOTAL_HOURS))
for t in range(TOTAL_HOURS):
    # Update each component's state
    for be in states:
        if not states[be]:                     # currently up
            if np.random.rand() < lambda_dict[be] * DT_HOURS:
                states[be] = True               # failure
        else:                                   # currently failed
            if np.random.rand() < mu_dict[be] * DT_HOURS:
                states[be] = False               # repair

    # Evaluate the fault tree expression with current states
    # (True means the basic event has occurred / component is failed)
    eval_dict = {be: states[be] for be in states}
    try:
        top = int(eval(py_expr, {}, eval_dict))
    except Exception as e:
        print(f"Error evaluating expression at hour {t}: {e}")
        top = 0
    top_event[t] = top

# ----------------------------------------------------------------------
# 5. Extract loss‑of‑supply intervals
# ----------------------------------------------------------------------
# Find transitions from 0→1 and 1→0
diff = np.diff(np.concatenate(([0], top_event, [0])))
starts = np.where(diff == 1)[0]          # indices where a block begins
ends   = np.where(diff == -1)[0]         # indices where a block ends (exclusive)

intervals = list(zip(starts, ends))
durations = [end - start for start, end in intervals]

total_interruptions = len(intervals)
total_duration_hours = sum(durations)
sim_hours = TOTAL_HOURS
sim_years = sim_hours / (365 * 24)

# ----------------------------------------------------------------------
# 6. Compute reliability indices
# ----------------------------------------------------------------------
SAIFI = total_interruptions / sim_years                     # interruptions/year
SAIDI = total_duration_hours / sim_years                    # hours/year
CAIDI = SAIDI / SAIFI if SAIFI > 0 else 0.0                 # hours/interruption

# For ENS and AENS we assume a constant load of 1 kW and one customer.
# Change these values if more information is available.
LOAD_KW = 1.0
NUM_CUSTOMERS = 1

ENS = total_duration_hours * LOAD_KW                        # kWh over simulation period
AENS = ENS / (sim_years * NUM_CUSTOMERS)                    # kWh/year per customer

availability = 1.0 - total_duration_hours / sim_hours

# ----------------------------------------------------------------------
# 7. Print results
# ----------------------------------------------------------------------
print("\n" + "="*60)
print("RELIABILITY INDICES (based on extracted data)")
print("="*60)
print(f"Simulation period: {sim_years:.2f} years")
print(f"Number of loss‑of‑supply events: {total_interruptions}")
print(f"Total interruption duration: {total_duration_hours:.2f} hours")
print(f"System availability: {availability:.6f}")
print("\nIndices (assuming constant load = 1 kW, 1 customer):")
print(f"  SAIFI = {SAIFI:.4f} interruptions/year")
print(f"  SAIDI = {SAIDI:.4f} hours/year")
print(f"  CAIDI = {CAIDI:.4f} hours/interruption")
print(f"  ENS   = {ENS:.2f} kWh (over {sim_years:.1f} years)")
print(f"  AENS  = {AENS:.4f} kWh/year")

# ----------------------------------------------------------------------
# 8. Optional: plot the loss‑of‑supply timeline (first few days)
# ----------------------------------------------------------------------
plot_days = 30                              # plot first 30 days
plot_hours = plot_days * 24
if plot_hours <= TOTAL_HOURS:
    plt.figure(figsize=(12, 3))
    plt.step(np.arange(plot_hours) / 24, top_event[:plot_hours],
             where='post', color='red', linewidth=1)
    plt.xlabel('Days')
    plt.ylabel('Loss of supply')
    plt.title(f'Loss‑of‑supply events (first {plot_days} days)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

print("\nSimulation finished.")