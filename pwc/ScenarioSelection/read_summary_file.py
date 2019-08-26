"""
Reads the PWC batch output file (usually Summary_SW.txt), creates percentiles for each scenario and selects a subset
based on a percentile and window
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

# This silences some error messages being raised by Pandas
pd.options.mode.chained_assignment = None

# Field information
hydro_groups = ['A', 'A/D', 'B', 'B/D', 'C', 'C/D', 'D']
pwc_id_fields = ['line_num', 'run_id']
pwc_run_id = ['run_id', 'pwc_scenario_id', 'rep']
pwc_durations = ['acute', 'chronic', 'cancer']
pwc_header = pwc_id_fields + pwc_durations
scenario_fields = ['scenario_id', 'state', 'soil_id', 'weather_grid', 'area', 'hydro_group']


def compute_percentiles(scenario_table):
    for field in pwc_durations:
        # Calculate percentiles
        data = scenario_table[['line_num', field, 'area']].sort_values(field).reset_index()
        weighted = ((np.cumsum(data.area) - 0.5 * data.area) / data.area.sum()) * 100
        # unweighted = ((data.index + 1) / data.shape[0]) * 100

        # Add new columns to scenario table
        new_header = ["line_num", "{}_%ile".format(field)]
        new_cols = pd.DataFrame(np.vstack((data.line_num, weighted)).T, columns=new_header)
        scenario_table = scenario_table.merge(new_cols, on="line_num")

    return scenario_table


def read_pwc_output(in_file):
    # Read the table, manually entering in the header (original header is tough to parse)
    table = pd.read_csv(in_file, names=pwc_header, delimiter='\s+')

    # Adjust line number by 2 so that header is not included
    table['line_num'] -= 1

    # Split the Batch Run ID field into constituent parts
    data = table.pop('run_id').str.split('_', expand=True)
    data.columns = pwc_run_id

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
    for duration, selection_pct in itertools.product(pwc_durations, selection_pcts):
        pct_field = duration + "_%ile"

        # Selects all scenarios within the window, or outside but with equal value
        lower = scenarios[pct_field] >= (selection_pct - window)
        upper = scenarios[pct_field] <= (selection_pct + window)
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


def plot_selection(scenarios, selection, plot_outfile):
    outfiles = []
    for field in pwc_durations:
        colors = np.array(['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'])
        fig, ax1 = plt.subplots()
        for group in sorted(scenarios.hydro_group.unique())[::-1]:
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
        outfiles.append(plot_outfile.format(field))
    return outfiles


def main(pwc_batch_file, pwc_summary_file, output_dir):
    """ Set run parameters here """
    selection_pcts = [50, 75, 90, 95]  # percentile for selection
    window = 0.1  # select scenarios within a range
    summary_outfile = os.path.join(output_dir, "test_summary.csv")
    selected_scenario_outfile = os.path.join(output_dir, "test_selection.csv")
    plot_outfile = os.path.join(output_dir, "plot_{}")
    """"""""""""""""""""""""""""""

    # Join the scenarios data with the computed concentrations
    scenario_ids = read_scenarios(pwc_batch_file)
    pwc_output = read_pwc_output(pwc_summary_file)
    print(pwc_output)
    exit()
    scenarios = scenario_ids.merge(pwc_output, on='line_num', how='left')

    # Calculate percentiles for test fields and append additional attributes
    scenarios = compute_percentiles(scenarios)

    # Select scenarios for each duration based on percentile, and write to file
    selection = select_scenarios(scenarios, selection_pcts, window)

    # Plot results and save tabular data to file
    outfiles = plot_selection(scenarios, selection, plot_outfile)
    scenarios.to_csv(summary_outfile, index=None)
    selection.to_csv(selected_scenario_outfile, index=None)
    return [summary_outfile, selected_scenario_outfile] + outfiles


if __name__ == "__main__":
    from paths import pwc_batch_file, pwc_summary_file, output_dir
    main(pwc_batch_file, pwc_summary_file, output_dir)
