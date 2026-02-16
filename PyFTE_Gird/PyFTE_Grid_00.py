import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# 1. Load data and discover components
# ----------------------------------------------------------------------
fault_log = pd.read_csv('fault_log.csv', parse_dates=['time'])
sensor_data = pd.read_csv('sensor_data.csv', index_col=0, parse_dates=True)

# Discover components from the fault log (unique values in 'component' column)
components = sorted(fault_log['component'].unique())
print(f"Discovered components: {components}")

# Top event from sensor data – we assume 'loss_of_supply' indicates system failure
top_event = 'loss_of_supply'
if top_event not in sensor_data.columns:
    raise ValueError(f"Column '{top_event}' not found in sensor data. Cannot determine top event.")

# Ensure all discovered components have a column in sensor_data
missing = [c for c in components if c not in sensor_data.columns]
if missing:
    print(f"Warning: The following components are not in sensor_data: {missing}")
    # Keep only those that exist
    components = [c for c in components if c in sensor_data.columns]

print(f"Using components: {components}")

# ----------------------------------------------------------------------
# 2. Build truth table from sensor data
# ----------------------------------------------------------------------
truth_table = sensor_data[components + [top_event]].copy()

def extract_cut_sets(df):
    """Extract all cut sets from rows where top event = 1."""
    basic = df.columns[:-1]
    te = df.columns[-1]
    cut_sets = []
    for _, row in df.iterrows():
        if row[te] == 1: # failed system
            cut_sets.append(set(basic[row[basic] == 0]))   # failed components
    return cut_sets

def get_minimal_cut_sets(cut_sets):
    """Keep only cut sets that are not supersets of any other."""
    cut_sets = sorted(cut_sets, key=lambda x: len(x))
    minimal = []
    for cs in cut_sets:
        if not any(cs.issuperset(mcs) for mcs in minimal):
            minimal.append(cs)
    return minimal

def build_boolean_expression(minimal_cut_sets):
    """Return sum‑of‑products expression."""
    return " + ".join(["·".join(sorted(mcs)) for mcs in minimal_cut_sets])

cut_sets = extract_cut_sets(truth_table)
minimal_cut_sets = get_minimal_cut_sets(cut_sets)

print("\n=== Learned Fault Tree Structure ===")
print("Minimal cut sets:")
for mcs in minimal_cut_sets:
    print(sorted(mcs))
print(f"Boolean expression: {top_event} = {build_boolean_expression(minimal_cut_sets)}")

# ----------------------------------------------------------------------
# 3. Build component event timelines from fault_log
# ----------------------------------------------------------------------
# Simulation time boundaries (from sensor_data index)
start_time = sensor_data.index[0]
end_time = sensor_data.index[-1] + pd.Timedelta(hours=1)   # last hour end

# Collect events per component (hardware failures + operational events)
component_events = {comp: [] for comp in components}
for _, row in fault_log.iterrows():
    comp = row['component']
    if comp in components:
        component_events[comp].append((row['time'], row['event']))

component_stats = {}
for comp in components:
    events = sorted(component_events[comp], key=lambda x: x[0])

    if not events:
        # No events recorded – component always up (from sensor_data)
        initial_state = True
        up_complete = []
        down_complete = []
        up_censored = (end_time - start_time).total_seconds() / 3600.0
        down_censored = 0.0
        n_failures = 0
    else:
        # Determine initial state from the first sensor reading
        initial_state = sensor_data.loc[start_time, comp] == 1
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

    component_stats[comp] = {
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
print("\n=== Fitted Exponential Parameters (MLE) ===")
for comp in components:
    stats = component_stats[comp]
    total_up = sum(stats['up_complete']) + stats['up_censored']
    total_down = sum(stats['down_complete']) + stats['down_censored']
    n_fail = stats['n_failures']
    n_rep = stats['n_repairs']

    lambda_hat = n_fail / total_up if total_up > 0 and n_fail > 0 else 0.0
    mu_hat = n_rep / total_down if total_down > 0 and n_rep > 0 else 0.0

    mttf = 1.0 / lambda_hat if lambda_hat > 0 else float('inf')
    mttr = 1.0 / mu_hat if mu_hat > 0 else float('inf')

    print(f"{comp}:")
    print(f"  Failures = {n_fail}, Total up = {total_up:.2f} h → λ = {lambda_hat:.6f} /h, MTTF = {mttf:.2f} h")
    print(f"  Repairs  = {n_rep}, Total down = {total_down:.2f} h → μ = {mu_hat:.6f} /h, MTTR = {mttr:.2f} h")

# # ----------------------------------------------------------------------
# # 5. (Optional) Weibull fit if lifelines is installed
# # ----------------------------------------------------------------------
# try:
#     from lifelines import WeibullFitter
#     print("\n=== Weibull Fits for Failure Times (with censoring) ===")
#     for comp in components:
#         stats = component_stats[comp]
#         durations = stats['up_complete'] + ([stats['up_censored']] if stats['up_censored'] > 0 else [])
#         event_observed = [1] * len(stats['up_complete']) + ([0] if stats['up_censored'] > 0 else [])
#         if len(durations) > 0:
#             wf = WeibullFitter().fit(durations, event_observed)
#             print(f"{comp}: shape ρ = {wf.rho_:.3f}, scale λ = {wf.lambda_:.3f}")
# except ImportError:
#     print("\nNote: Install 'lifelines' for Weibull/lognormal fits with censoring.")