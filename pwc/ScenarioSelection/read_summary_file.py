"""
Reads the PWC batch output file (usually Summary_SW.txt), creates percentiles for each scenario and selects a subset
based on a percentile and window
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from paths import pwc_batch_file, pwc_summary_file, summary_outfile, selected_scenario_outfile, plot_outfile

# This silences some error messages being raised by Pandas
pd.options.mode.chained_assignment = None

# Field information
hydro_groups = ['A', 'A/D', 'B', 'B/D', 'C', 'C/D', 'D']
pwc_id_fields = ['line_num', 'run_id']
run_id_fields = ['run_id', 'pwc_scenario_id', 'rep']
pwc_out_fields = ['peak', '1-day', 'year', 'overall', '4-day', '21-day', '60-day', '90-day', 'pw_peak', 'pw_21']
pwc_header = pwc_id_fields + pwc_out_fields
scenario_fields = ['scenario_id', 'state', 'soil_id', 'weather_grid', 'area', 'hydro_group']
id_fields = ['line_num', 'run_id', 'scenario_id_pwc']
test_fields = ['1-day', '4-day', '21-day', '60-day', '90-day', 'peak', 'year', 'overall']


def compute_percentiles(scenario_table):
    for field in test_fields:
        # Calculate percentiles
        data = scenario_table[[field, 'area']].sort_values(field).reset_index()
        weighted = ((np.cumsum(data.area) - 0.5 * data.area) / data.area.sum()) * 100
        # unweighted = ((data.index + 1) / data.shape[0]) * 100
        # Add new columns to scenario table
        scenario_table[field + "_%ile"] = weighted
    return scenario_table


def read_pwc_output(in_file):
    # Read the table, manually entering in the header (original header is tough to parse)
    table = pd.read_csv(in_file, skiprows=1, names=pwc_header, delimiter='\s+')

    # Adjust line number by 2 so that header is not included
    table['line_num'] -= 2

    # Split the Batch Run ID field into constituent parts
    data = table.pop('run_id').str.split('_', expand=True)
    data.columns = ['run_id', 'pwc_scenario_id', 'rep']

    return pd.concat([data, table], axis=1)


def read_scenarios(scenario_table):
    scenarios = pd.read_csv(scenario_table, dtype={'area': np.int64})[scenario_fields]
    scenarios['line_num'] = scenarios.index + 1
    return scenarios


def select_scenarios(scenarios, selection_pcts, window):
    # Designate the lower and upper bounds for the percentile selection
    window /= 2  # half below, half above
    # Select scenarios for each of the durations and combine
    all_selected = []
    for duration, selection_pct in itertools.product(test_fields, selection_pcts):

        # Selects all scenarios within the window, or outside but with equal value
        lower = scenarios[duration + "_%ile"] >= (selection_pct - window)
        upper = scenarios[duration + "_%ile"] <= (selection_pct + window)
        selected = scenarios[lower & upper]
        min_val, max_val = selected[duration].min(), selected[duration].max()
        selected = (scenarios[duration] >= min_val) & (scenarios[duration] <= max_val)

        # Set new fields
        selection = scenarios[selected][scenario_fields]
        selection['target'] = selection_pct
        selection['duration'] = duration
        selection['percentile'] = scenarios[selected][duration + "_%ile"]
        selection['conc'] = scenarios[selected][duration]
        all_selected.append(selection)
    all_selected = pd.concat(all_selected, axis=0)

    return all_selected.sort_values(['duration', 'area'], ascending=[True, False])


def plot_selection(test_fields, scenarios, selection):
    for field in test_fields:
        colors = np.array(['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'])
        fig, ax1 = plt.subplots()
        for group in sorted(scenarios.hydro_group.unique()):
            sel = scenarios[scenarios.hydro_group == group]
            concs, pctiles = sel[[field, '{}_%ile'.format(field)]].values.T
            ax1.scatter(concs, pctiles, s=10, c=colors[group - 1], label=hydro_groups[group - 1])
        concs, pctiles, col = \
            selection.loc[selection.duration == field, ['conc', 'percentile', 'hydro_group']].values.T
        ax1.set_label('Soil Group')
        plt.xlabel('Concentration (μg/L)', fontsize=12)
        plt.ylabel('Percentile', fontsize=12)
        plt.legend(loc='upper left')
        ax1.scatter(concs, pctiles, s=100, c=colors[col.astype(np.int8) - 1], label="Selected")
        plt.savefig(plot_outfile.format(field), dpi=600)


def main():
    """ Set run parameters here """
    selection_pcts = [50, 75, 90, 95]  # percentile for selection
    window = 0.1  # select scenarios within a range
    """"""""""""""""""""""""""""""

    # Join the scenarios data with the computed concentrations
    scenario_ids = read_scenarios(pwc_batch_file)
    pwc_output = read_pwc_output(pwc_summary_file)
    scenarios = scenario_ids.merge(pwc_output, on='line_num', how='left')

    # Calculate percentiles for test fields and append additional attributes
    scenarios = compute_percentiles(scenarios)

    # Select scenarios for each duration based on percentile, and write to file
    selection = select_scenarios(scenarios, selection_pcts, window)

    # Plot results and save tabular data to file
    plot_selection(test_fields, scenarios, selection)
    scenarios.to_csv(summary_outfile, index=None)
    selection.to_csv(selected_scenario_outfile, index=None)


main()
