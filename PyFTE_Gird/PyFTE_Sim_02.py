"""
PyFTE_Sim_02.py (Final, cleaned)

Discrete‑event simulation of microgrid reliability using:
  - fault_tree_expression.csv   (Boolean expression for loss_of_supply)
  - distribution_parameters.csv (realistic exponential/Weibull parameters)

Produces:
  - SAIFI, SAIDI, CAIDI, ENS, AENS (repairable case)
  - High‑resolution plots of:
        * Availability over time (with 95% CI using margin)
        * Unavailability (log scale)
        * Reliability (survival probability, repairable case)
  - Component unavailability ranking
  - Importance measures (Birnbaum, Criticality, Fussell‑Vesely) printed
  - Non‑repairable MTTF printed (but no plot)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import re
import itertools
from scipy.special import gamma

# Set high default DPI for all plots
plt.rcParams['figure.dpi'] = 300

# ----------------------------------------------------------------------
# 1. Load input data
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 1. Load input data
# ----------------------------------------------------------------------
try:
    ft_df = pd.read_csv('fault_tree_expression.csv')
    dist_df = pd.read_csv('distribution_parameters.csv')
except FileNotFoundError as e:
    print("Error: missing CSV file(s).")
    raise e

print("\nComponent mapping (Basic Event ID -> Component):")
for _, row in dist_df.iterrows():
    print(f"  {row['ID']} = {row['Basic Event']}")

top_event_name = ft_df['Top_Event'].iloc[0]
fault_expr = ft_df['Factored_Expression'].iloc[0]

# Dictionaries for exponential and Weibull parameters
lambda_dict = {}
mu_dict = {}
mttr_dict = {}
weibull_dict = {}

for _, row in dist_df.iterrows():
    be_id = row['ID']
    mttr = row['MTTR_h']
    exp_lambda = row['Exp_λ_per_h']
    weib_rho = row['Weibull_ρ']
    weib_lam = row['Weibull_λ_h']

    if weib_rho != 'N/A' and weib_lam != 'N/A':
        weibull_dict[be_id] = {'shape': float(weib_rho), 'scale': float(weib_lam)}
        mu_dict[be_id] = 1.0 / mttr if mttr > 0 else 0.0
        mttr_dict[be_id] = mttr
    else:
        lambda_dict[be_id] = exp_lambda
        mu_dict[be_id] = 1.0 / mttr if mttr > 0 else 0.0
        mttr_dict[be_id] = mttr

basic_events = sorted(set(lambda_dict.keys()) | set(weibull_dict.keys()))

# ----------------------------------------------------------------------
# 2. Extract minimal cut sets from Boolean expression
# ----------------------------------------------------------------------
def get_minimal_cut_sets_from_expression(expr):
    be_ids = sorted(set(re.findall(r'BE\d+', expr)))
    n = len(be_ids)
    py_expr = expr.replace('·', ' and ').replace('+', ' or ')
    truth = []
    for bits in itertools.product([0, 1], repeat=n):
        state = {be: bool(bits[i]) for i, be in enumerate(be_ids)}
        if eval(py_expr, {}, state):
            failed = {be for be, val in state.items() if val}
            truth.append(failed)
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
# 3. Repairable discrete‑event simulation (exponential + Weibull)
# ----------------------------------------------------------------------
def simulate_reliability_repairable(minimal_cut_sets, exp_rates, weibull_params,
                                    repair_times, T, dt=1.0, N_SIM=1000, rng_seed=42):
    np.random.seed(rng_seed)

    be_list = sorted(set(exp_rates.keys()) | set(weibull_params.keys()))
    be_to_idx = {be: i for i, be in enumerate(be_list)}
    n_be = len(be_list)
    mttr = repair_times

    time_grid = np.arange(0, T + dt, dt)
    n_out = len(time_grid)
    top_event_states = np.zeros((N_SIM, n_out), dtype=int)

    for sim in range(N_SIM):
        state = np.zeros(n_be, dtype=int)
        next_event_time = np.zeros(n_be)

        # Initial failure times
        for i, be in enumerate(be_list):
            if be in weibull_params:
                shape = weibull_params[be]['shape']
                scale = weibull_params[be]['scale']
                next_event_time[i] = np.random.weibull(shape) * scale if (shape>0 and scale>0) else np.inf
            else:
                lam = exp_rates.get(be, 0.0)
                next_event_time[i] = np.random.exponential(1.0/lam) if lam>0 else np.inf

        sys_down = False
        current_time = 0.0
        out_idx = 0

        while current_time < T and out_idx < n_out:
            next_t = np.min(next_event_time)
            if next_t == np.inf:
                next_t = T
            else:
                next_t = min(next_t, T)

            while out_idx < n_out and time_grid[out_idx] <= next_t:
                top_event_states[sim, out_idx] = 1 if sys_down else 0
                out_idx += 1
            if out_idx >= n_out:
                break

            current_time = next_t
            for i, be in enumerate(be_list):
                if abs(next_event_time[i] - current_time) < 1e-9:
                    if state[i] == 0:       # failure
                        state[i] = 1
                        if mttr[be] < np.inf:
                            next_event_time[i] = current_time + np.random.exponential(mttr[be])
                        else:
                            next_event_time[i] = np.inf
                    else:                   # repair
                        state[i] = 0
                        if be in weibull_params:
                            shape = weibull_params[be]['shape']
                            scale = weibull_params[be]['scale']
                            if shape>0 and scale>0:
                                next_event_time[i] = current_time + np.random.weibull(shape) * scale
                            else:
                                next_event_time[i] = np.inf
                        else:
                            lam = exp_rates.get(be, 0.0)
                            if lam>0:
                                next_event_time[i] = current_time + np.random.exponential(1.0/lam)
                            else:
                                next_event_time[i] = np.inf

            sys_down = any(all(state[be_to_idx[be]] == 1 for be in cut) for cut in minimal_cut_sets)

        while out_idx < n_out:
            top_event_states[sim, out_idx] = 1 if sys_down else 0
            out_idx += 1

    availability = 1.0 - np.mean(top_event_states, axis=0)
    ever_failed = np.maximum.accumulate(top_event_states, axis=1)
    reliability = 1.0 - np.mean(ever_failed, axis=0)

    return {'time_grid': time_grid,
            'availability': availability,
            'reliability': reliability,
            'top_event_states': top_event_states}

# ----------------------------------------------------------------------
# 4. Run repairable simulation and compute indices
# ----------------------------------------------------------------------
SIM_YEARS = 5
T_mission = SIM_YEARS * 365 * 24
N_SIM = 5000        # adjust for speed vs accuracy
DT_OUT = 1.0

print(f"\nRunning repairable DES for {SIM_YEARS} years ({T_mission} hours) with {N_SIM} replications...")
t_start = time.time()
res = simulate_reliability_repairable(minimal_cut_sets, lambda_dict, weibull_dict,
                                      mttr_dict, T_mission, dt=DT_OUT, N_SIM=N_SIM, rng_seed=42)
t_des = time.time() - t_start
print(f"DES finished in {t_des:.2f} seconds")

# SAIFI, SAIDI, CAIDI, ENS, AENS
interruptions = []
down_hours = []
states = res['top_event_states']
for sim in range(N_SIM):
    s = states[sim, :]
    diff = np.diff(np.concatenate(([0], s, [0])))
    n_int = np.sum(diff == 1)
    down = np.sum(s) * DT_OUT
    interruptions.append(n_int)
    down_hours.append(down)

mean_int = np.mean(interruptions)
mean_down = np.mean(down_hours)
sim_years = T_mission / (365*24)

SAIFI = mean_int / sim_years
SAIDI = mean_down / sim_years
CAIDI = SAIDI / SAIFI if SAIFI>0 else 0.0
LOAD_KW = 1.0
NUM_CUST = 1
ENS = mean_down * LOAD_KW
AENS = ENS / (sim_years * NUM_CUST)

print("\n" + "="*60)
print("RELIABILITY INDICES (repairable case)")
print("="*60)
print(f"SAIFI = {SAIFI:.4f} interruptions/year")
print(f"SAIDI = {SAIDI:.4f} hours/year")
print(f"CAIDI = {CAIDI:.4f} hours/interruption")
print(f"ENS   = {ENS:.2f} kWh (over {sim_years:.1f} years)")
print(f"AENS  = {AENS:.4f} kWh/year")
print(f"System unavailability: {mean_down / T_mission:.6f}")

# ----------------------------------------------------------------------
# 5. High‑resolution plots (availability, unavailability, reliability)
# ----------------------------------------------------------------------
time_grid = res['time_grid']
avail = res['availability']
unavail = 1.0 - avail

# Compute confidence interval using standard error (margin)
avail_per_sim = 1.0 - states
avail_mean = avail_per_sim.mean(axis=0)
avail_std = avail_per_sim.std(axis=0, ddof=1)
avail_se = avail_std / np.sqrt(N_SIM)
z = 1.96  # 95% confidence
avail_lo = avail_mean - z * avail_se
avail_hi = avail_mean + z * avail_se

# Figure 1: Availability with margin CI
plt.figure(figsize=(14, 6), dpi=900)
plt.plot(time_grid / 8760, avail_mean, linewidth=2, color='blue', label='Mean availability')
plt.fill_between(time_grid / 8760, avail_lo, avail_hi, alpha=0.2, color='blue', label='95% CI (margin)')
plt.xlabel('Time (years)', fontsize=12)
plt.ylabel('Availability', fontsize=12)
plt.title(f'System Availability over Time (repairable, {N_SIM} replications)', fontsize=14)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Figure 2: Unavailability (log scale)
plt.figure(figsize=(14, 6), dpi=900)
plt.semilogy(time_grid / 8760, unavail, linewidth=2, color='red')
plt.xlabel('Time (years)', fontsize=12)
plt.ylabel('Unavailability (log scale)', fontsize=12)
plt.title('System Unavailability over Time', fontsize=14)
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.show()

# Figure 3: Reliability (survival probability, repairable case)
plt.figure(figsize=(14, 6), dpi=900)
plt.plot(time_grid / 8760, res['reliability'], linewidth=2, color='green')
plt.xlabel('Time (years)', fontsize=12)
plt.ylabel('Reliability (no failure probability)', fontsize=12)
plt.title('System Reliability Function (repairable case)', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# 6. Component unavailability (steady‑state approximation)
# ----------------------------------------------------------------------
comp_unavail = {}
for be in basic_events:
    if be in weibull_dict:
        shape = weibull_dict[be]['shape']
        scale = weibull_dict[be]['scale']
        mttf_approx = scale * gamma(1 + 1/shape) if shape>0 else np.inf
        mttr = mttr_dict[be]
        unav = mttr / (mttf_approx + mttr) if np.isfinite(mttf_approx) else 1.0
    else:
        lam = lambda_dict.get(be, 0.0)
        mu = 1.0 / mttr_dict[be] if mttr_dict[be]>0 else 0.0
        unav = lam / (lam + mu) if (lam+mu)>0 else 0.0
    comp_unavail[be] = unav

comp_df = pd.DataFrame(list(comp_unavail.items()), columns=['Component', 'Unavailability']).set_index('Component')
comp_df = comp_df.sort_values('Unavailability', ascending=False)
print("\nComponent unavailability (steady‑state):")
print(comp_df.round(6))

# Bar plot (high resolution)
plt.figure(figsize=(10, 6), dpi=900)
comp_sorted = comp_df.sort_values('Unavailability', ascending=True)
plt.barh(comp_sorted.index, comp_sorted['Unavailability'].values, color='steelblue')
plt.xlabel('Unavailability (fraction of time down)', fontsize=12)
plt.title('Component Unavailability', fontsize=14)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# 7. Non‑repairable reliability (no plot, only MTTF printed)
# ----------------------------------------------------------------------
def comp_failure_prob(be, t):
    if be in weibull_dict:
        shape = weibull_dict[be]['shape']
        scale = weibull_dict[be]['scale']
        return 1.0 - np.exp(-(t/scale)**shape)
    else:
        lam = lambda_dict.get(be, 0.0)
        return 1.0 - np.exp(-lam * t)

def top_failure_prob(t, cut_sets, prob_func):
    probs = [np.prod([prob_func(be, t) for be in cut]) for cut in cut_sets]
    prob_top = sum(probs)
    n = len(cut_sets)
    for i in range(n):
        for j in range(i+1, n):
            inter = cut_sets[i] | cut_sets[j]
            p_int = np.prod([prob_func(be, t) for be in inter])
            prob_top -= p_int
    return min(1.0, max(0.0, prob_top))

t_max = T_mission
t_fine = np.linspace(0, t_max, 500)
R_nr = np.array([1.0 - top_failure_prob(t, minimal_cut_sets, comp_failure_prob) for t in t_fine])
mttf_nr = np.trapezoid(R_nr, t_fine)
print(f"\nNon‑repairable MTTF (up to {t_max/8760:.1f} years): {mttf_nr:.2f} hours ({mttf_nr/8760:.2f} years)")
# ----------------------------------------------------------------------
# 8. Importance measures (structural, marginal, criticality, Fussell‑Vesely)
# ----------------------------------------------------------------------
def compute_structural_importance(cut_sets, be_list):
    """
    Compute structural importance I_Phi for each basic event.
    Formula: I_Φ(BE) = 1 - ∏_{i: BE ∈ MCS_i} [1 - 1/2^(N_i - 1)]
    where N_i is the number of basic events in cut set i.
    """
    struct_imp = {}
    for be in be_list:
        product = 1.0
        for cut in cut_sets:
            if be in cut:
                n = len(cut)
                product *= (1.0 - 1.0 / (2 ** (n - 1)))
        struct_imp[be] = 1.0 - product
    return struct_imp

MISSION_TIME_HOURS = 8760   # 1 year (can be changed)
print(f"\nImportance analysis for mission time T = {MISSION_TIME_HOURS} hours (no repairs)")

# Component failure probabilities at mission time
p_be = {be: comp_failure_prob(be, MISSION_TIME_HOURS) for be in basic_events}
Q_top = top_failure_prob(MISSION_TIME_HOURS, minimal_cut_sets, comp_failure_prob)

# Structural importance (topology‑based)
structural = compute_structural_importance(minimal_cut_sets, basic_events)

# Marginal (Birnbaum), Criticality, Fussell‑Vesely
birnbaum = {}
criticality = {}
fussell_vesely = {}

for be in basic_events:
    # Birnbaum: sum over cuts containing be of product of other probabilities
    dQ = 0.0
    for cut in minimal_cut_sets:
        if be in cut:
            others = [c for c in cut if c != be]
            prod_others = np.prod([p_be[c] for c in others])
            dQ += prod_others
    birnbaum[be] = dQ
    criticality[be] = (p_be[be] * dQ) / Q_top if Q_top > 0 else 0.0
    # Fussell‑Vesely: fraction of Q from cuts containing be
    q_with = 0.0
    for cut in minimal_cut_sets:
        if be in cut:
            q_with += np.prod([p_be[c] for c in cut])
    fussell_vesely[be] = q_with / Q_top if Q_top > 0 else 0.0

# Build DataFrame with all four measures, sorted by BE number
imp_df = pd.DataFrame({
    'Structural': structural,
    'Marginal (Birnbaum)': birnbaum,
    'Criticality': criticality,
    'Fussell-Vesely': fussell_vesely
}).sort_index(key=lambda x: x.str.extract(r'(\d+)', expand=False).astype(int))

print("\nImportance measures at T =", MISSION_TIME_HOURS, "hours")
print(imp_df.round(6))