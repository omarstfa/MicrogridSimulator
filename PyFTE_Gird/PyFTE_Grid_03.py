import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# 1. Load data and discover basic events
# ----------------------------------------------------------------------
fault_log = pd.read_csv('fault_log.csv', parse_dates=['time'])
state_data = pd.read_csv('state_data_with_override.csv', index_col=0, parse_dates=True)

print("\n" + "="*60)
print("Extracted Events")
print("="*60)

# Discover basic events from the fault log (unique values in 'component' column)
basic_events = sorted(fault_log['component'].unique())
print(f"Discovered basic events: {basic_events}")

# Top event from state data – we assume 'loss_of_supply' indicates system failure
top_event = 'loss_of_supply'
if top_event not in state_data.columns:
    raise ValueError(f"Column '{top_event}' not found in state data. Cannot determine top event.")

# Ensure all discovered basic events have a column in state_data
missing = [c for c in basic_events if c not in state_data.columns]
if missing:
    print(f"Warning: The following basic events are not in state_data: {missing}")
    # Keep only those that exist
    basic_events = [c for c in basic_events if c in state_data.columns]

print(f"Using basic events: {basic_events}")

# ----------------------------------------------------------------------
# 2. Build truth table from state data
# ----------------------------------------------------------------------
truth_table = state_data[basic_events + [top_event]].copy()

def extract_cut_sets(df):
    """Extract all cut sets from rows where top event = 1."""
    basic = df.columns[:-1]
    te = df.columns[-1]
    cut_sets = []
    for _, row in df.iterrows():
        if row[te] == 1: # failed system
            cut_sets.append(set(basic[row[basic] == 0]))   # failed basic events
    return cut_sets

def get_minimal_cut_sets(cut_sets):
    """Keep only cut sets that are not supersets of any other."""
    cut_sets = sorted(cut_sets, key=lambda x: len(x))
    minimal = []
    for cs in cut_sets:
        if not any(cs.issuperset(mcs) for mcs in minimal):
            minimal.append(cs)
    return minimal

def make_be_map(minimal_cut_sets):
    """Assign BE# labels to each unique basic event, sorted alphabetically."""
    all_events = sorted({ev for mcs in minimal_cut_sets for ev in mcs})
    be_map    = {ev: f"BE{i+1}" for i, ev in enumerate(all_events)}
    be_legend = {f"BE{i+1}": ev for i, ev in enumerate(all_events)}
    return be_map, be_legend

def factor_sop(terms):
    """
    Recursively factor a sum-of-products (list of frozensets of BE# labels)
    using greedy single-literal extraction.
    Returns a factored string expression using · for AND and + for OR.
    """
    from collections import Counter

    if not terms:
        return "0"
    if len(terms) == 1:
        t = sorted(terms[0])
        return "·".join(t) if t else "1"

    # Count occurrences of each literal across all terms
    lit_count = Counter(lit for term in terms for lit in term)
    if not lit_count:
        return "1"

    # Pick literal with highest count; break ties alphabetically
    best_lit = sorted(lit_count, key=lambda x: (-lit_count[x], x))[0]

    if lit_count[best_lit] < 2:
        # No further factoring possible — emit flat SOP
        parts = ["·".join(sorted(t)) for t in sorted(terms, key=lambda x: (len(x), sorted(x)))]
        return " + ".join(parts)

    with_lit    = [t - {best_lit} for t in terms if best_lit in t]
    without_lit = [t              for t in terms if best_lit not in t]

    inner = factor_sop(with_lit)
    # Wrap in parentheses only if the inner expression is a sum
    factored = f"{best_lit}·({inner})" if " + " in inner else f"{best_lit}·{inner}"

    if without_lit:
        return f"{factored} + {factor_sop(without_lit)}"
    return factored

def build_flat_sop(minimal_cut_sets, label_map=None):
    """Return a flat (un-factored) sum-of-products expression.
    If label_map is provided, basic event names are replaced with its values."""
    parts = []
    for mcs in minimal_cut_sets:
        literals = sorted(label_map[c] if label_map else c for c in mcs)
        parts.append("\u00b7".join(literals))
    return " + ".join(parts)

def build_factored_expr(minimal_cut_sets, label_map=None):
    """Return a factored Boolean expression.
    If label_map is provided, basic event names are replaced with its values."""
    if label_map:
        terms = [frozenset(label_map[c] for c in mcs) for mcs in minimal_cut_sets]
    else:
        terms = [frozenset(mcs) for mcs in minimal_cut_sets]
    return factor_sop(terms)

cut_sets = extract_cut_sets(truth_table)
minimal_cut_sets = get_minimal_cut_sets(cut_sets)
be_map, be_legend = make_be_map(minimal_cut_sets)

print("\n" + "="*60)
print("Learned Fault Tree Structure")
print("="*60)
print("Minimal cut sets (basic event names):")
for mcs in minimal_cut_sets:
    print(sorted(mcs))

# --- (A) Original flat SOP with basic event names ---
flat_names = build_flat_sop(minimal_cut_sets)
print("\n" + "-"*60)
print("(A) Original flat SOP (basic event names)")
print("-"*60)
print(f"  {top_event} = {flat_names}")

# --- (B) Factored SOP with basic event names ---
factored_names = build_factored_expr(minimal_cut_sets)
print("\n" + "-"*60)
print("(B) Factored expression (basic event names)")
print("-"*60)
print(f"  {top_event} = {factored_names}")

# --- (C) Factored SOP with BE# labels ---
legend_lines = [f"  {lbl} = {name}" for lbl, name in sorted(be_legend.items())]
print("\n" + "-"*60)
print("(C) Factored expression (BE# labels)")
print("-"*60)
print("Basic Events (BE) Legend:")
print("\n".join(legend_lines))
factored_be = build_factored_expr(minimal_cut_sets, label_map=be_map)
print(f"\n  {top_event} = {factored_be}")

# ----------------------------------------------------------------------
# 3. Build basic event timelines from fault_log
# ----------------------------------------------------------------------
# Simulation time boundaries (from state_data index)
start_time = state_data.index[0]
end_time = state_data.index[-1] + pd.Timedelta(hours=1)   # last hour end

# Collect fault events per basic event (hardware failures + operational events)
be_timelines = {event: [] for event in basic_events}
for _, row in fault_log.iterrows():
    be = row['component']
    if be in basic_events:
        be_timelines[be].append((row['time'], row['event']))

be_stats = {}
for event in basic_events:
    events = sorted(be_timelines[event], key=lambda x: x[0])

    if not events:
        # No events recorded – basic event always healthy (from state_data)
        initial_state = True
        up_complete = []
        down_complete = []
        up_censored = (end_time - start_time).total_seconds() / 3600.0
        down_censored = 0.0
        n_failures = 0
    else:
        # Determine initial state from the first state reading
        initial_state = state_data.loc[start_time, event] == 1
        current_state = initial_state
        last_time = start_time
        up_complete = []
        down_complete = []

        for ev_time, ev_type in events:
            delta = (ev_time - last_time).total_seconds() / 3600.0
            if current_state:
                # Up period ends with a failure event
                if ev_type in ['FAILURE', 'SOC_DEPLETED', 'OUTPUT_ZERO']:
                    up_complete.append(delta)
                    current_state = False
            else:
                # Down period ends with a repair event
                if ev_type in ['REPAIR', 'SOC_RECOVERED', 'OUTPUT_ACTIVE']:
                    down_complete.append(delta)
                    current_state = True
            last_time = ev_time

        # Final interval to end_time (right‑censored)
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
# 4. Exponential fit (MLE with right‑censoring)
# ----------------------------------------------------------------------
print("\n" + "="*60)
print("Fitted Exponential Parameters (MLE)")
print("="*60)
for event in basic_events:
    stats = be_stats[event]
    total_up = sum(stats['up_complete']) + stats['up_censored']
    total_down = sum(stats['down_complete']) + stats['down_censored']
    n_fail = stats['n_failures']
    n_rep = stats['n_repairs']

    lambda_hat = n_fail / total_up if total_up > 0 and n_fail > 0 else 0.0
    mu_hat = n_rep / total_down if total_down > 0 and n_rep > 0 else 0.0

    mttf = 1.0 / lambda_hat if lambda_hat > 0 else float('inf')
    mttr = 1.0 / mu_hat if mu_hat > 0 else float('inf')

    print(f"{event}:")
    print(f"  Failures = {n_fail}, Total up = {total_up:.2f} h → λ = {lambda_hat:.6f} /h, MTTF = {mttf:.2f} h")
    print(f"  Repairs  = {n_rep}, Total down = {total_down:.2f} h → μ = {mu_hat:.6f} /h, MTTR = {mttr:.2f} h")

# ----------------------------------------------------------------------
# 5. (Optional) Weibull fit if lifelines is installed
# ----------------------------------------------------------------------
try:
    from lifelines import WeibullFitter
    print("\n" + "="*60)
    print("Weibull Fits for Failure Times (with censoring)")
    print("="*60)
    for event in basic_events:
        stats = be_stats[event]
        raw_durations      = stats['up_complete'] + ([stats['up_censored']] if stats['up_censored'] > 0 else [])
        raw_event_observed = [1] * len(stats['up_complete']) + ([0] if stats['up_censored'] > 0 else [])

        # lifelines requires strictly positive durations.
        # Zero durations arise when two events share the same timestamp.
        pairs = [(d, e) for d, e in zip(raw_durations, raw_event_observed) if d > 0]
        n_dropped = len(raw_durations) - len(pairs)

        if not pairs:
            print(f"{event}: no positive durations — skipping Weibull fit")
            continue

        durations, event_observed = zip(*pairs)
        wf = WeibullFitter().fit(list(durations), list(event_observed))
        note = f"  ({n_dropped} zero-duration interval(s) dropped)" if n_dropped > 0 else ""
        print(f"{event}: shape ρ = {wf.rho_:.3f}, scale λ = {wf.lambda_:.3f}{note}")
except ImportError:
    print("\nNote: Install 'lifelines' for Weibull/lognormal fits with censoring.")
    
#%% Summary

# ----------------------------------------------------------------------
# 6. Important learned parameters (fault tree + distributions)
# ----------------------------------------------------------------------
print("\n" + "="*60)
print("EXPORT: Learned System Parameters")
print("="*60)

# Prepare export data structure
export_data = {
    'fault_tree': {
        'top_event': top_event,
        'factored_expression_names': factored_names,
        'factored_expression_labels': factored_be,
        'minimal_cut_sets': [sorted(list(mcs)) for mcs in minimal_cut_sets],
        'basic_events_legend': be_legend
    },
    'exponential_params': {},
    'weibull_params': {}
}

# Collect exponential and Weibull parameters
for event in basic_events:
    stats = be_stats[event]
    total_up = sum(stats['up_complete']) + stats['up_censored']
    total_down = sum(stats['down_complete']) + stats['down_censored']
    n_fail = stats['n_failures']
    n_rep = stats['n_repairs']
    
    lambda_hat = n_fail / total_up if total_up > 0 and n_fail > 0 else 0.0
    mu_hat = n_rep / total_down if total_down > 0 and n_rep > 0 else 0.0
    
    export_data['exponential_params'][event] = {
        'failure_rate': round(lambda_hat, 6),
        'repair_rate': round(mu_hat, 6),
        'mttf_hours': round(1.0/lambda_hat, 2) if lambda_hat > 0 else float('inf'),
        'mttr_hours': round(1.0/mu_hat, 2) if mu_hat > 0 else float('inf'),
        'failures_observed': n_fail,
        'repairs_observed': n_rep
    }

# Add Weibull parameters if lifelines was available
try:
    from lifelines import WeibullFitter
    
    for event in basic_events:
        stats = be_stats[event]
        raw_durations = stats['up_complete'] + ([stats['up_censored']] if stats['up_censored'] > 0 else [])
        raw_event_observed = [1] * len(stats['up_complete']) + ([0] if stats['up_censored'] > 0 else [])
        
        pairs = [(d, e) for d, e in zip(raw_durations, raw_event_observed) if d > 0]
        
        if pairs:
            durations, event_observed = zip(*pairs)
            wf = WeibullFitter().fit(list(durations), list(event_observed))
            export_data['weibull_params'][event] = {
                'shape_rho': round(wf.rho_, 3),
                'scale_lambda': round(wf.lambda_, 3)
            }
except:
    export_data['weibull_params'] = "Weibull fitting not available (install lifelines)"

# Print condensed export summary
print("\nFAULT TREE (Factored):")
print(f"  {top_event} = {factored_be}")

print("\nBASIC EVENTS - Distribution Parameters:")
print("  " + "="*90)
print("  ID      Basic Event     Occurrences  MTTF(h)  MTTR(h)   Exp_λ(/h)      Weibull_ρ     Weibull_λ(h)")
print("  " + "-"*90)
for be_id, be_name in sorted(be_legend.items()):
    exp = export_data['exponential_params'][be_name]
    weib = export_data['weibull_params'].get(be_name, {}) if isinstance(export_data['weibull_params'], dict) else {}
    
    exp_rate = f"{exp['failure_rate']:.6f}"
    weib_rho = f"{weib.get('shape_rho', 'N/A'):.3f}" if isinstance(weib.get('shape_rho'), float) else "N/A"
    weib_lambda = f"{weib.get('scale_lambda', 'N/A'):.1f}" if isinstance(weib.get('scale_lambda'), float) else "N/A"
    
    print(f"  {be_id:4}    {be_name:12}   {exp['failures_observed']:8d}   {exp['mttf_hours']:7.1f}  {exp['mttr_hours']:7.1f}    {exp_rate:>10}   {weib_rho:>10}   {weib_lambda:>12}")

print("  " + "="*90)
print("\n  Exp_λ: Exponential failure rate")
print("  Weibull_ρ: Weibull shape parameter (ρ<1: infant mortality, ρ=1: random failures, ρ>1: wear-out)")
print("  Weibull_λ: Weibull scale parameter (characteristic life)")

#%% Export CSV

# ----------------------------------------------------------------------
# 7. Save to CSV files
# ----------------------------------------------------------------------
import csv

# Save fault tree expression to CSV
with open('fault_tree_expression.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Top_Event', 'Factored_Expression'])
    writer.writerow([top_event, factored_be])

# Save distribution parameters to CSV
with open('distribution_parameters.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ID', 'Basic Event', 'Occurrences', 'MTTF_h', 'MTTR_h', 
                     'Exp_λ_per_h', 'Weibull_ρ', 'Weibull_λ_h'])
    for be_id, be_name in sorted(be_legend.items()):
        exp = export_data['exponential_params'][be_name]
        weib = export_data['weibull_params'].get(be_name, {}) if isinstance(export_data['weibull_params'], dict) else {}
        writer.writerow([
            be_id, be_name, exp['failures_observed'], exp['mttf_hours'], exp['mttr_hours'],
            exp['failure_rate'],
            weib.get('shape_rho', 'N/A'),
            weib.get('scale_lambda', 'N/A')
        ])

print("\nFiles saved: fault_tree_expression.csv, distribution_parameters.csv")