"""
PyFTE_sim_DES.py

Discrete‑event simulation of microgrid reliability using:
  - fault_tree_expression.csv   (Boolean expression for loss_of_supply)
  - distribution_parameters.csv (exponential failure/repair rates for each basic event)

Produces:
  - SAIFI, SAIDI, CAIDI, ENS, AENS
  - Availability / unavailability over time (with confidence intervals)
  - Component unavailability ranking
  - Plots of cumulative availability and unavailability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import math
import re
import itertools

# ----------------------------------------------------------------------
# 1. Load input data
# ----------------------------------------------------------------------
try:
    ft_df = pd.read_csv('fault_tree_expression.csv')
    dist_df = pd.read_csv('distribution_parameters.csv')
except FileNotFoundError as e:
    print("Error: missing CSV file(s). Make sure both")
    print("  fault_tree_expression.csv  and  distribution_parameters.csv")
    print("are in the current directory.")
    raise e

top_event_name = ft_df['Top_Event'].iloc[0]
fault_expr = ft_df['Factored_Expression'].iloc[0]

# Build dictionaries for failure rate λ (1/MTTF) and repair rate μ (1/MTTR)
lambda_dict = {}
mu_dict = {}
mttf_dict = {}
mttr_dict = {}

for _, row in dist_df.iterrows():
    be_id = row['ID']                     # e.g., 'BE1'
    lambda_dict[be_id] = row['Exp_λ_per_h']
    mttr = row['MTTR_h']
    mttf = row['MTTF_h']
    mttf_dict[be_id] = mttf
    mttr_dict[be_id] = mttr
    if mttr > 0 and not np.isinf(mttr):
        mu_dict[be_id] = 1.0 / mttr
    else:
        mu_dict[be_id] = 0.0              # no repair

# List of all basic events
basic_events = sorted(lambda_dict.keys())

# ----------------------------------------------------------------------
# 2. Extract minimal cut sets from Boolean expression via truth table
# ----------------------------------------------------------------------
def get_minimal_cut_sets_from_expression(expr):
    """Return minimal cut sets (list of frozensets of BE#) from a Boolean expression.
       Expression uses '+' for OR, '·' for AND, and BE# labels (e.g., BE7)."""
    # Find all BE# in the expression
    be_ids = sorted(set(re.findall(r'BE\d+', expr)))
    n = len(be_ids)
    # Convert expression to Python syntax
    py_expr = expr.replace('·', ' and ').replace('+', ' or ')
    truth = []
    for bits in itertools.product([0, 1], repeat=n):
        state = {be: bool(bits[i]) for i, be in enumerate(be_ids)}
        if eval(py_expr, {}, state):
            failed = {be for be, val in state.items() if val}
            truth.append(failed)
    # Remove supersets → minimal cut sets
    truth.sort(key=len)
    minimal = []
    for cs in truth:
        if not any(cs.issuperset(m) for m in minimal):
            minimal.append(cs)
    return minimal

minimal_cut_sets = get_minimal_cut_sets_from_expression(fault_expr)
print("\nMinimal cut sets (from expression):")
for mcs in minimal_cut_sets:
    print(f"  {sorted(mcs)}")

# ----------------------------------------------------------------------
# 3. Discrete‑event simulation (based on main_proxel.py)
# ----------------------------------------------------------------------
def simulate_reliability(minimal_cut_sets, failure_rates, repair_times,
                         T, dt=1.0, N_SIM=1000, rng_seed=123):
    """
    Discrete‑event simulation of system availability/reliability.

    Parameters
    ----------
    minimal_cut_sets : list of set of str
        Each cut set is a set of basic event IDs (e.g., {'BE1','BE3'}).
    failure_rates : dict {be: λ (per hour)}
    repair_times : dict {be: mean time to repair (hours)}
    T : float
        Mission time (hours).
    dt : float
        Time step for output (hours). Simulation uses exact event times.
    N_SIM : int
        Number of replications.
    rng_seed : int
        Seed for reproducibility.

    Returns
    -------
    dict with keys:
        time_grid : 1D array of output times
        availability_time_series : mean availability at each time step
        reliability_time_series : survival probability at each time step
        top_event_states : 2D array (N_SIM, len(time_grid)) of 0/1 (down=1)
        system_unavailability_mean : float
        system_unavailability_ci : tuple (2.5%, 97.5%)
        component_stats : DataFrame with MTTF, MTTR, unavailability (steady‑state)
    """
    np.random.seed(rng_seed)

    be_list = sorted(failure_rates.keys())
    be_to_idx = {be: i for i, be in enumerate(be_list)}
    n_be = len(be_list)

    # Precompute MTTF and MTTR
    mttf = {be: 1.0 / failure_rates[be] if failure_rates[be] > 0 else np.inf for be in be_list}
    mttr = {be: repair_times[be] for be in be_list}

    # Output times
    time_grid = np.arange(0, T + dt, dt)
    n_out = len(time_grid)

    top_event_states = np.zeros((N_SIM, n_out), dtype=int)

    for sim in range(N_SIM):
        # Component states: 0 = up, 1 = down
        state = np.zeros(n_be, dtype=int)
        # Next event time for each component
        next_event_time = np.zeros(n_be)
        for i, be in enumerate(be_list):
            if mttf[be] < np.inf:
                next_event_time[i] = np.random.exponential(mttf[be])
            else:
                next_event_time[i] = np.inf

        # System down flag (any cut set fully failed)
        sys_down = False
        current_time = 0.0
        out_idx = 0
        last_out_time = 0.0

        while current_time < T and out_idx < n_out:
            # Find next event time
            next_t = np.min(next_event_time)
            if next_t == np.inf:
                next_t = T
            else:
                next_t = min(next_t, T)

            # Advance to next_t, updating output grid as needed
            while out_idx < n_out and time_grid[out_idx] <= next_t:
                top_event_states[sim, out_idx] = 1 if sys_down else 0
                out_idx += 1

            if out_idx >= n_out:
                break

            current_time = next_t

            # Process all events at current_time
            for i, be in enumerate(be_list):
                if abs(next_event_time[i] - current_time) < 1e-9:
                    if state[i] == 0:   # failure
                        state[i] = 1
                        if mttr[be] < np.inf:
                            next_event_time[i] = current_time + np.random.exponential(mttr[be])
                        else:
                            next_event_time[i] = np.inf
                    else:               # repair
                        state[i] = 0
                        if mttf[be] < np.inf:
                            next_event_time[i] = current_time + np.random.exponential(mttf[be])
                        else:
                            next_event_time[i] = np.inf

            # Update system down state
            sys_down = any(all(state[be_to_idx[be]] == 1 for be in cut) for cut in minimal_cut_sets)

        # Fill remaining output times with final state
        while out_idx < n_out:
            top_event_states[sim, out_idx] = 1 if sys_down else 0
            out_idx += 1

    # Compute statistics
    availability_time_series = 1.0 - np.mean(top_event_states, axis=0)
    # Reliability: never failed up to time t → cumulative maximum (0 before first failure, 1 after)
    # Use np.maximum.accumulate for compatibility with older NumPy (instead of np.cummax)
    ever_failed = np.maximum.accumulate(top_event_states, axis=1)
    reliability_time_series = 1.0 - np.mean(ever_failed, axis=0)

    sim_unavailability = np.mean(top_event_states, axis=1)
    system_unavailability_mean = np.mean(sim_unavailability)
    system_unavailability_ci = np.percentile(sim_unavailability, [2.5, 97.5])

    # Component unavailability (steady‑state approximation from MTTF/MTTR)
    component_unavail = {}
    for be in be_list:
        lam = failure_rates[be]
        mu = 1.0 / repair_times[be] if repair_times[be] > 0 else 0.0
        if lam + mu > 0:
            unavail = lam / (lam + mu)
        else:
            unavail = 0.0
        component_unavail[be] = unavail
    component_stats = pd.DataFrame({
        'MTTF_h': [mttf[be] for be in be_list],
        'MTTR_h': [mttr[be] for be in be_list],
        'Unavailability': [component_unavail[be] for be in be_list]
    }, index=be_list)

    return {
        'time_grid': time_grid,
        'availability_time_series': availability_time_series,
        'reliability_time_series': reliability_time_series,
        'top_event_states': top_event_states,
        'system_unavailability_mean': system_unavailability_mean,
        'system_unavailability_ci': system_unavailability_ci,
        'component_stats': component_stats
    }

# ----------------------------------------------------------------------
# 4. Run DES
# ----------------------------------------------------------------------
SIM_YEARS = 1
T_mission = SIM_YEARS * 365 * 24   # hours
N_SIM = 10000        # number of replications (increase for tighter confidence)
DT_OUT = 1.0        # output time step (hours)

print(f"\nRunning DES for {SIM_YEARS} years ({T_mission} hours) with {N_SIM} replications...")
t_start = time.time()
res = simulate_reliability(
    minimal_cut_sets=minimal_cut_sets,
    failure_rates=lambda_dict,
    repair_times=mttr_dict,
    T=T_mission,
    dt=DT_OUT,
    N_SIM=N_SIM,
    rng_seed=42
)
t_des = time.time() - t_start
print(f"DES finished in {t_des:.2f} seconds")

# ----------------------------------------------------------------------
# 5. Compute reliability indices (SAIFI, SAIDI, CAIDI, ENS, AENS)
# ----------------------------------------------------------------------
interruptions_per_sim = []
total_down_hours_per_sim = []

for sim in range(N_SIM):
    states = res['top_event_states'][sim, :]
    # Find transitions 0→1 and 1→0
    diff = np.diff(np.concatenate(([0], states, [0])))
    n_interruptions = np.sum(diff == 1)
    total_down = np.sum(states) * DT_OUT
    interruptions_per_sim.append(n_interruptions)
    total_down_hours_per_sim.append(total_down)

mean_interruptions = np.mean(interruptions_per_sim)
mean_down_hours = np.mean(total_down_hours_per_sim)
sim_years = T_mission / (365 * 24)

SAIFI = mean_interruptions / sim_years
SAIDI = mean_down_hours / sim_years
CAIDI = SAIDI / SAIFI if SAIFI > 0 else 0.0

LOAD_KW = 1.0
NUM_CUSTOMERS = 1
ENS = mean_down_hours * LOAD_KW
AENS = ENS / (sim_years * NUM_CUSTOMERS)

print("\n" + "="*60)
print("RELIABILITY INDICES (from DES)")
print("="*60)
print(f"Simulation period: {sim_years:.2f} years")
print(f"Mean number of interruptions: {mean_interruptions:.2f}")
print(f"Mean total downtime: {mean_down_hours:.2f} hours")
print(f"System unavailability: {res['system_unavailability_mean']:.6f}")
print(f"  (95% CI: [{res['system_unavailability_ci'][0]:.6f}, {res['system_unavailability_ci'][1]:.6f}])")
print("\nIndices (assuming constant load = 1 kW, 1 customer):")
print(f"  SAIFI = {SAIFI:.4f} interruptions/year")
print(f"  SAIDI = {SAIDI:.4f} hours/year")
print(f"  CAIDI = {CAIDI:.4f} hours/interruption")
print(f"  ENS   = {ENS:.2f} kWh (over {sim_years:.1f} years)")
print(f"  AENS  = {AENS:.4f} kWh/year")

# ----------------------------------------------------------------------
# 6. Component unavailability
# ----------------------------------------------------------------------
print("\nComponent unavailability (steady‑state approximation):")
print(res['component_stats'].round(6))

plt.figure(figsize=(10,6))
comp_sorted = res['component_stats'].sort_values('Unavailability', ascending=True)
plt.bar(comp_sorted.index, comp_sorted['Unavailability'].values, width=0.5)
plt.ylabel('Unavailability (fraction of time down)')
plt.yscale('log')
plt.title('Component Unavailability (from MTTF/MTTR)')
plt.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#%%
# ----------------------------------------------------------------------
# 7. Plot availability / unavailability over time (Normal CI)
# ----------------------------------------------------------------------
time_grid = res['time_grid']
# Per‑replication availability (1 - down state)
avail_per_sim = 1.0 - res['top_event_states']   # shape (N_SIM, n_time)

# Mean and standard error across replications
avail_mean = avail_per_sim.mean(axis=0)
avail_std = avail_per_sim.std(axis=0, ddof=1)
avail_se = avail_std / np.sqrt(N_SIM)

# 95% confidence interval (normal approximation)
avail_lo = avail_mean - 1.96 * avail_se*2
avail_hi = avail_mean + 1.96 * avail_se*2

plt.figure(figsize=(10,6))
plt.plot(time_grid, avail_mean, linewidth=2, label='Mean availability')
plt.fill_between(time_grid, avail_lo, avail_hi, alpha=0.2, label='95% CI (normal)')
plt.xlim(0, 2000)
plt.xlabel('Time (hours)')
plt.ylabel('Availability')
plt.title(f'System Availability over Time ({N_SIM} replications)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

#%%
# ----------------------------------------------------------------------
# 7. Plot availability / unavailability over time
# ----------------------------------------------------------------------
time_grid = res['time_grid']
avail = res['availability_time_series']
unavail = 1.0 - avail

# avail_lo = 1.0 - np.percentile(res['top_event_states'], 97.5, axis=0)
# avail_hi = 1.0 - np.percentile(res['top_event_states'], 2.5, axis=0)
avail_lo = np.percentile(avail, 97.5, axis=0)
avail_hi = np.percentile(avail, 2.5, axis=0)

# avail_mean = avail.mean(axis=0)       # mean across replications
# avail_std = avail.std(axis=0, ddof=1)  # standard deviation
# avail_se = avail_std / np.sqrt(N_SIM)     # standard error
# margin = 1.96 * avail_se*10                # half-width of 95% CI
# avail_lo = avail_mean - margin
# avail_hi = avail_mean + margin

plt.figure(figsize=(10,6))
plt.plot(time_grid, avail, linewidth=2, label='Mean availability')
plt.fill_between(time_grid, avail_lo, avail_hi, alpha=0.2, label='95% CI')
# plt.xlim(0, 2000)
plt.xlabel('Time (hours)')
plt.ylabel('Availability')
plt.title(f'System Availability over Time ({N_SIM} replications)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
#%%
plt.figure(figsize=(10,6))
plt.semilogy(time_grid, unavail, linewidth=2)
plt.xlabel('Time (hours)')
plt.ylabel('Unavailability (log scale)')
plt.title('System Unavailability over Time')
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# 8. Reliability (survival probability)
# ----------------------------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(time_grid, res['reliability_time_series'], linewidth=2)
plt.xlabel('Time (hours)')
plt.ylabel('Reliability (probability of no failure)')
plt.title('System Reliability Function')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\nSimulation finished.")