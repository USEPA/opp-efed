"""
Reads the PWC batch output file (usually Summary_SW.txt), creates percentiles for each scenario and selects a subset
based on a percentile and window
"""
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

# This silences some error messages being raised by Pandas
pd.options.mode.chained_assignment = None

# Field information
hydro_groups = ['A', 'A/D', 'B', 'B/D', 'C', 'C/D', 'D']
pwc_run_id = ['run_id', 'pwc_scenario_id', 'rep']
pwc_id_fields = ['line_num', 'run_id']
pwc_durations = ['acute', 'chronic', 'cancer']
pwc_header = pwc_id_fields + pwc_durations
scenario_fields = ['scenario_id', 'state', 'soil_id', 'weather_grid', 'area', 'hydro_group']
percentile_field = "{}_%ile"


def compute_percentiles(scenario_table, weighted):
    for field in pwc_durations:
        # Calculate percentiles
        data = scenario_table[['line_num', field, 'area']].sort_values(field).reset_index()
        if weighted:
            percentiles = ((np.cumsum(data.area) - 0.5 * data.area) / data.area.sum()) * 100
        else:
            percentiles = ((data.index + 1) / data.shape[0]) * 100

        # Add new columns to scenario table
        new_header = ["line_num", percentile_field.format(field)]
        new_cols = pd.DataFrame(np.vstack((data.line_num, percentiles)).T, columns=new_header)
        scenario_table = scenario_table.merge(new_cols, on="line_num")

    return scenario_table


def plot_selection(scenarios, selection, plot_outfile):
    outfiles = []
    for field in pwc_durations:
        colors = np.array(['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'])
        fig, ax1 = plt.subplots()
        for group in sorted(scenarios.hydro_group.unique())[::-1]:
            sel = scenarios[scenarios.hydro_group == group]
            concentrations, percentiles = sel[[field, percentile_field.format(field)]].values.T
            ax1.scatter(concentrations, percentiles, s=10, c=colors[group - 1], label=hydro_groups[group - 1])
        concentrations, percentiles, hsg = \
            selection.loc[selection.duration == field, ['concentration', 'percentile', 'hydro_group']].values.T
        ax1.set_label('Soil Group')
        plt.xlabel('Concentration (μg/L)', fontsize=12)
        plt.ylabel('Percentile', fontsize=12)
        plt.legend(loc='upper left')
        ax1.scatter(concentrations, percentiles, s=100, c=colors[hsg.astype(np.int8) - 1], label="Selected")
        plt.savefig(plot_outfile.format(field), dpi=600)
        outfiles.append(plot_outfile.format(field))
    return outfiles


def read_pwc_output(in_file):
    print(in_file)
    # Read the table, manually entering in the header (original header is tough to parse)
    table = pd.read_csv(in_file, names=pwc_header, delimiter=r'\s+')
    print(table)
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
        # Selects all scenarios within the window, or outside but with equal value
        lower = scenarios[percentile_field.format(duration)] >= (selection_pct - window)
        upper = scenarios[percentile_field.format(duration)] <= (selection_pct + window)
        selected = scenarios[lower & upper]
        min_val, max_val = selected[duration].min(), selected[duration].max()
        selected = (scenarios[duration] >= min_val) & (scenarios[duration] <= max_val)

        # Set new fields
        selection = scenarios[selected][scenario_fields]
        selection['target'] = selection_pct
        selection['duration'] = duration
        selection['percentile'] = scenarios[selected][percentile_field.format(duration)]
        selection['concentration'] = scenarios[selected][duration]
        all_selected.append(selection)

    all_selected = pd.concat(all_selected, axis=0)

    return all_selected.sort_values(['duration', 'area'], ascending=[True, False])


def initialize_output(pwc_infile, output_path, weighted):
    scenario_file = os.path.basename(pwc_infile)
    file_format = re.compile('(r[\dNSEWUL]{1,3})_\d{1,3}_([A-Za-z\s]+?)_')
    try:
        region, crop = re.match(file_format, scenario_file).groups()
        results_dir = os.path.join(output_path, "{}_{}_summary_files".format(region, crop))
        tag = "{}_{}{}_".format(region, crop, "_unweighted" if not weighted else "")
    except AttributeError:
        print("Can't read region or crop from '{}'".format(scenario_file))
        results_dir = os.path.join(output_path, "pwc_output_summary_files")
        tag = ""
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    return [os.path.join(results_dir, tag + t) for t in ['summary.csv', 'selected.csv', '{}.png']]


def main(pwc_infile, pwc_outfile, output_path, selection_pcts, window, weighted=True):
    summary_outfile, selected_scenario_outfile, plot_outfile = \
        initialize_output(pwc_infile, output_path, weighted)

    # Join the scenarios data with the computed concentrations
    scenarios = read_scenarios(pwc_infile)
    pwc_output = read_pwc_output(pwc_outfile)
    scenarios = scenarios.merge(pwc_output, on='line_num', how='left')

    # Calculate percentiles for test fields and append additional attributes
    scenarios = compute_percentiles(scenarios, weighted)

    # Select scenarios for each duration based on percentile, and write to file
    selection = select_scenarios(scenarios, selection_pcts, window)

    # Plot results and save tabular data to file
    outfiles = plot_selection(scenarios, selection, plot_outfile)
    scenarios.to_csv(summary_outfile, index=None)
    selection.to_csv(selected_scenario_outfile, index=None)
    return [summary_outfile, selected_scenario_outfile] + outfiles


if __name__ == "__main__":
    # Paths
    pwc_summary_file = r"J:\opp-efed\pwc\ScenarioSelection\Input\Samples\BatchOutputVVWM.txt"  # the table used as input for the pwc run
    pwc_batch_file = r"J:\opp-efed\pwc\ScenarioSelection\Input\Samples\r1_1_Corn_fix.csv"  # the output file from the pwc run
    output_dir = "Output"
    selection_percentiles = [50, 75, 90, 95]  # percentile for selection
    selection_window = 0.1  # select scenarios within a range
    area_weighting = True
    main(pwc_batch_file, pwc_summary_file, output_dir, selection_percentiles, selection_window, area_weighting)
