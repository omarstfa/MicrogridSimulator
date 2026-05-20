import pandas as pd
import numpy as np
import csv
import re
import itertools

# ----------------------------------------------------------------------
# 1. Load data and discover basic events
# ----------------------------------------------------------------------
SCALE = 10  # Must match the scaling factor used in microgrid_sim_12

fault_log = pd.read_csv('fault_log.csv', parse_dates=['time'])
state_data = pd.read_csv('state_data_with_override.csv', index_col=0, parse_dates=True)

print("\n" + "="*60)
print("Extracted Events (scaled simulation)")
print("="*60)

basic_events = sorted(fault_log['component'].unique())
print(f"Discovered basic events: {basic_events}")

top_event = 'loss_of_supply'
if top_event not in state_data.columns:
    raise ValueError(f"Column '{top_event}' not found in state data.")

missing = [c for c in basic_events if c not in state_data.columns]
if missing:
    print(f"Warning: missing columns {missing}")
    basic_events = [c for c in basic_events if c in state_data.columns]

print(f"Using basic events: {basic_events}")

# ----------------------------------------------------------------------
# 2. Build truth table and minimal cut sets (same as PyFTE_Grid_03)
# ----------------------------------------------------------------------
truth_table = state_data[basic_events + [top_event]].copy()

def extract_cut_sets(df):
    basic = df.columns[:-1]
    te = df.columns[-1]
    cut_sets = []
    for _, row in df.iterrows():
        if row[te] == 1:
            cut_sets.append(set(basic[row[basic] == 0]))
    return cut_sets

def get_minimal_cut_sets(cut_sets):
    cut_sets = sorted(cut_sets, key=lambda x: len(x))
    minimal = []
    for cs in cut_sets:
        if not any(cs.issuperset(mcs) for mcs in minimal):
            minimal.append(cs)
    return minimal

def make_be_map(minimal_cut_sets):
    all_events = sorted({ev for mcs in minimal_cut_sets for ev in mcs})
    be_map = {ev: f"BE{i+1}" for i, ev in enumerate(all_events)}
    be_legend = {f"BE{i+1}": ev for i, ev in enumerate(all_events)}
    return be_map, be_legend

def factor_sop(terms):
    from collections import Counter
    if not terms:
        return "0"
    if len(terms) == 1:
        t = sorted(terms[0])
        return "·".join(t) if t else "1"
    lit_count = Counter(lit for term in terms for lit in term)
    if not lit_count:
        return "1"
    best_lit = sorted(lit_count, key=lambda x: (-lit_count[x], x))[0]
    if lit_count[best_lit] < 2:
        parts = ["·".join(sorted(t)) for t in sorted(terms, key=lambda x: (len(x), sorted(x)))]
        return " + ".join(parts)
    with_lit = [t - {best_lit} for t in terms if best_lit in t]
    without_lit = [t for t in terms if best_lit not in t]
    inner = factor_sop(with_lit)
    factored = f"{best_lit}·({inner})" if " + " in inner else f"{best_lit}·{inner}"
    if without_lit:
        return f"{factored} + {factor_sop(without_lit)}"
    return factored

def build_factored_expr(minimal_cut_sets, label_map=None):
    if label_map:
        terms = [frozenset(label_map[c] for c in mcs) for mcs in minimal_cut_sets]
    else:
        terms = [frozenset(mcs) for mcs in minimal_cut_sets]
    return factor_sop(terms)

cut_sets = extract_cut_sets(truth_table)
minimal_cut_sets = get_minimal_cut_sets(cut_sets)
be_map, be_legend = make_be_map(minimal_cut_sets)

print("\nMinimal cut sets (basic event names):")
for mcs in minimal_cut_sets:
    print(sorted(mcs))

factored_be = build_factored_expr(minimal_cut_sets, label_map=be_map)

# ----------------------------------------------------------------------
# 3. Build basic event timelines (same as before)
# ----------------------------------------------------------------------
start_time = state_data.index[0]
end_time = state_data.index[-1] + pd.Timedelta(hours=1)

be_timelines = {event: [] for event in basic_events}
for _, row in fault_log.iterrows():
    be = row['component']
    if be in basic_events:
        be_timelines[be].append((row['time'], row['event']))

be_stats = {}
for event in basic_events:
    events = sorted(be_timelines[event], key=lambda x: x[0])
    if not events:
        initial_state = True
        up_complete = []
        down_complete = []
        up_censored = (end_time - start_time).total_seconds() / 3600.0
        down_censored = 0.0
        n_failures = 0
    else:
        initial_state = state_data.loc[start_time, event] == 1
        current_state = initial_state
        last_time = start_time
        up_complete = []
        down_complete = []
        for ev_time, ev_type in events:
            delta = (ev_time - last_time).total_seconds() / 3600.0
            if current_state:
                if ev_type in ['FAILURE', 'SOC_DEPLETED', 'OUTPUT_ZERO']:
                    up_complete.append(delta)
                    current_state = False
            else:
                if ev_type in ['REPAIR', 'SOC_RECOVERED', 'OUTPUT_ACTIVE']:
                    down_complete.append(delta)
                    current_state = True
            last_time = ev_time
        final_delta = (end_time - last_time).total_seconds() / 3600.0
        if current_state:
            up_censored = final_delta
            down_censored = 0.0
        else:
            up_censored = 0.0
            down_censored = final_delta
        n_failures = len(up_complete)
    be_stats[event] = {
        'up_complete': up_complete,
        'down_complete': down_complete,
        'up_censored': up_censored,
        'down_censored': down_censored,
        'n_failures': n_failures,
        'n_repairs': len(down_complete)
    }

# ----------------------------------------------------------------------
# 4. Exponential fit (scaled rates) then correct by dividing by SCALE
# ----------------------------------------------------------------------
print("\n" + "="*60)
print("Fitted Exponential Parameters (scaled) and corrected (÷SCALE)")
print("="*60)

corrected_exp_params = {}
for event in basic_events:
    stats = be_stats[event]
    total_up = sum(stats['up_complete']) + stats['up_censored']
    total_down = sum(stats['down_complete']) + stats['down_censored']
    n_fail = stats['n_failures']
    n_rep = stats['n_repairs']
    lambda_scaled = n_fail / total_up if total_up > 0 and n_fail > 0 else 0.0
    mu_scaled = n_rep / total_down if total_down > 0 and n_rep > 0 else 0.0
    lambda_correct = lambda_scaled / SCALE
    mu_correct = mu_scaled / SCALE
    mttf_correct = 1.0 / lambda_correct if lambda_correct > 0 else float('inf')
    mttr_correct = 1.0 / mu_correct if mu_correct > 0 else float('inf')
    corrected_exp_params[event] = {
        'failure_rate': lambda_correct,
        'repair_rate': mu_correct,
        'mttf_hours': mttf_correct,
        'mttr_hours': mttr_correct,
        'failures_observed': n_fail,
        'repairs_observed': n_rep
    }
    print(f"{event}: λ_scaled = {lambda_scaled:.6f} → λ_real = {lambda_correct:.6e} /h, MTTF = {mttf_correct:.2f} h")
    print(f"       μ_scaled = {mu_scaled:.6f} → μ_real = {mu_correct:.6f} /h, MTTR = {mttr_correct:.2f} h")

# ----------------------------------------------------------------------
# 5. Weibull fit on scaled data with improved numerical stability
# ----------------------------------------------------------------------
corrected_weibull = {}
try:
    from lifelines import WeibullFitter
    from lifelines.exceptions import ConvergenceError

    print("\n" + "="*60)
    print("Weibull fits (scaled) and corrected (scale × SCALE)")
    print("="*60)

    # To avoid convergence issues, we temporarily multiply durations by a factor
    # that brings them to a reasonable range (e.g., near 1), then correct later.
    NUM_STAB_FACTOR = 1000.0   # can be tuned

    for event in basic_events:
        stats = be_stats[event]
        raw_durations = stats['up_complete'] + ([stats['up_censored']] if stats['up_censored'] > 0 else [])
        raw_event_observed = [1] * len(stats['up_complete']) + ([0] if stats['up_censored'] > 0 else [])
        pairs = [(d, e) for d, e in zip(raw_durations, raw_event_observed) if d > 0]
        if not pairs:
            print(f"{event}: no positive durations — skipping Weibull")
            continue

        durations, event_observed = zip(*pairs)
        # Scale durations up for numerical stability
        durations_scaled = [d * NUM_STAB_FACTOR for d in durations]

        try:
            wf = WeibullFitter().fit(list(durations_scaled), list(event_observed))
            shape = wf.rho_
            scale_scaled_stab = wf.lambda_
            # Reverse the temporary scaling
            scale_scaled = scale_scaled_stab / NUM_STAB_FACTOR
            scale_correct = scale_scaled * SCALE
            corrected_weibull[event] = {'shape_rho': shape, 'scale_lambda': scale_correct}
            print(f"{event}: shape ρ = {shape:.3f}, scale_scaled = {scale_scaled:.3f} → scale_real = {scale_correct:.3f} h")
        except ConvergenceError:
            print(f"Warning: Weibull fitting did not converge for {event}. Keeping exponential only.")
            corrected_weibull[event] = {}   # empty, will be treated as exponential later

except ImportError:
    print("\nNote: Install 'lifelines' for Weibull fitting. Using exponential only.")
    corrected_weibull = {}

# ----------------------------------------------------------------------
# 6. Export corrected distribution parameters and fault tree expression
# ----------------------------------------------------------------------
be_id_from_name = {name: be_id for be_id, name in be_legend.items()}

with open('distribution_parameters.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ID', 'Basic Event', 'Occurrences', 'MTTF_h', 'MTTR_h',
                     'Exp_λ_per_h', 'Weibull_ρ', 'Weibull_λ_h'])
    for be_id, be_name in sorted(be_legend.items()):
        exp = corrected_exp_params[be_name]
        weib = corrected_weibull.get(be_name, {})
        weib_rho = weib.get('shape_rho', 'N/A')
        weib_lam = weib.get('scale_lambda', 'N/A')
        writer.writerow([
            be_id, be_name, exp['failures_observed'],
            f"{exp['mttf_hours']:.2f}", f"{exp['mttr_hours']:.2f}",
            f"{exp['failure_rate']:.6e}",
            weib_rho, weib_lam
        ])

with open('fault_tree_expression.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Top_Event', 'Factored_Expression'])
    writer.writerow([top_event, factored_be])

print("\n" + "="*60)
print("EXPORT: Learned System Parameters (CORRECTED)")
print("="*60)
print(f"Fault tree expression: {top_event} = {factored_be}")
print("\nCorrected distribution parameters saved to 'distribution_parameters.csv'")
print("Fault tree expression saved to 'fault_tree_expression.csv'")