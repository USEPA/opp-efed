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
file_index = ['cdl_name', 'cdl', 'region', 'koc', 'scenario_file', 'pwc_outfile']
percentile_field = "{}_%ile"
concentration_field = "{}_koc{}"  # duration, koc
scenario_id_field = 'pwc_scenario_id'

# Parameters
kocs = [10, 1000, 10000]
selection_percentiles = [50, 75, 90, 95]  # percentile for selection
selection_window = 0.5  # select scenarios within a range
area_weighting = True

# Paths
root_dir = r"C:\Users\Jhook\Environmental Protection Agency (EPA)\Spatial Aquatic Model (SAM) Development - Documents\PWC Scenarios\PFTTT_ScenariosProject\ScenarioBatchFiles"
scenario_dir = os.path.join(root_dir, "Corn", "PWC_Field_Scenarios_Corn")
pwc_output_dir = os.path.join(root_dir, "Corn", "Comprehensive_Batch_Summary Files")
scenario_format = re.compile("(r[\dNSEWUL]{1,3})_(\d{1,3})_([A-Za-z\s]+?)[\.\_]")
pwc_output_format = re.compile("(r[\dNSEWUL]{1,3})_(\d{1,3})_([A-Za-z\s]+?)_koc(\d{2,5})")
output_dir = os.path.join(root_dir, "Corn", "Postprocessed")


def combine_tables(files):
    all_tables = []

    for _, (cdl_name, cdl, region, koc, scenario_file, pwc_outfile) in files.iterrows():
        # print(f"Reading tables for Region {region} {cdl_name}, Koc {koc}")
        scenarios = read_infiles(scenario_file, pwc_outfile)
        scenarios['region'] = region
        scenarios['cdl'] = cdl
        scenarios['cdl_name'] = cdl_name
        scenarios['koc'] = koc
        all_tables.append(scenarios)
    full_table = pd.concat(all_tables, axis=0)
    # TODO - there are rows with nan values... why?
    full_table = full_table.dropna()
    return full_table


def compute_percentiles(scenarios, fields, area_weight=True):
    duration_fields = [f for f in scenarios.columns if f in fields]
    for field in duration_fields:
        # Calculate percentiles
        scenarios = scenarios.sort_values(field)
        if area_weight:
            percentiles = ((np.cumsum(scenarios.area) - 0.5 * scenarios.area) / scenarios.area.sum()) * 100
        else:
            percentiles = ((scenarios.index + 1) / scenarios.shape[0]) * 100
        scenarios[percentile_field.format(field)] = percentiles
    return scenarios


def detect_params(pwc_outfile):
    pwc_outfile = os.path.basename(pwc_outfile)
    pattern = re.compile("(r[\dNSEWUL]{1,3})_(\d{1,3})_([A-Za-z\s]+?)_[Kk]oc(\d{2,5})")
    match = re.match(pattern, pwc_outfile)
    try:
        return match.groups()
    except AttributeError:
        print("Unable to parse region, cdl class and Koc from {}".format(pwc_outfile))
        return '', '', ''


def fetch_files():
    # Initialize dictionary with values of [scenario path, pwc output path]
    scenarios = {}

    # Find scenario files
    for f in os.listdir(scenario_dir):
        match = re.match(scenario_format, f)
        if match:
            region, cdl, cdl_name = match.groups()
            scenarios[(region, cdl)] = os.path.join(scenario_dir, f)

    # Find PWC output files
    all_files = []
    for f in os.listdir(pwc_output_dir):
        match = re.match(pwc_output_format, f)
        if match:
            region, cdl, cdl_name, koc = match.groups()
            scenario_file = scenarios.get((region, cdl))
            pwc_outfile = os.path.join(pwc_output_dir, f)
            all_files.append([cdl_name, cdl, region, koc, scenario_file, pwc_outfile])
    all_files = pd.DataFrame(all_files, columns=file_index).sort_values(by=file_index[1:4])

    return all_files


def initialize_output(region=None, crop=None, koc=None):
    if all((region, crop, koc)):
        tag = "{}_{}_{}_".format(region, crop, koc)
        outfiles = ['summary.csv', 'selected.csv', '{}']
    else:
        tag = 'national_'
        outfiles = ['{}_all_{}_']
    results_dir = os.path.join(output_dir, tag + "summary_files")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return [os.path.join(results_dir, tag + t) for t in outfiles]


def national_analysis(table, field1, field2):
    # TODO - crop needs to be specified here
    plot_outfile = initialize_output()[0]
    table = table[[scenario_id_field, 'duration', 'koc', 'conc', 'area']]
    fields = sorted(table[field2].unique())
    scenario_areas = table[[scenario_id_field, 'area']].drop_duplicates()
    for selection_val, subset in table.groupby(field1):
        # TODO - there are scenarios with duplicate values... why??
        subset = subset.groupby([scenario_id_field, field2]).mean().reset_index()
        subset = subset.pivot(index=scenario_id_field, columns=field2, values='conc')
        subset = subset.merge(scenario_areas, on=scenario_id_field, how='left')
        # TODO - there are rows with nan values... why?
        subset = subset.dropna()
        subset = compute_percentiles(subset, fields)
        plot(subset, fields, combined_label=field2.capitalize(), labels=fields,
             plot_outfile=plot_outfile.format(selection_val, field2), xmax=120)


def read_infiles(pwc_infile, pwc_outfile):
    # Join the scenarios data with the computed concentrations
    errors = []
    infiles = []
    for infile, read_function in ((pwc_infile, read_scenarios), (pwc_outfile, read_pwc_output)):
        try:
            infiles.append(read_function(infile))
        except TypeError:
            errors.append('File {} is invalid'.format(os.path.basename(infile)))
        except ValueError:
            errors.append('File not found')

    if any(errors):
        for error in errors:
            print(error)
        return
    else:
        scenarios, pwc_output = infiles
        return scenarios.merge(pwc_output, on='line_num', how='left')


def read_pwc_output(in_file):
    # Read the table, manually entering in the header (original header is tough to parse)
    table = pd.read_csv(in_file, names=pwc_header, delimiter=r'\s+')

    # Adjust line number so that header is not included
    table['line_num'] -= 1

    # Split the Batch Run ID field into constituent parts
    data = table.pop('run_id').str.split('_', expand=True)
    data.columns = pwc_run_id

    table = pd.concat([data, table], axis=1)
    table = table.melt(id_vars=[f for f in table.columns if not f in pwc_durations], value_vars=pwc_durations,
                       var_name='duration', value_name='conc')

    return table


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


def streamline_selection(selection):
    """ Ranges of selections don't plot well. Take the ones with the largest areas for plotting """
    selection = selection.reset_index()
    selection = selection.loc[selection.groupby(['target', 'duration'])['area'].idxmax()]
    return selection


def plot(scenarios, fields, plot_outfile=None, selection=None, combined=True,
         labels=None, combined_label=None, individual_label=None, selection_label=None, xmax=None):
    # TODO - adjust for outliers in display xmax
    def initialize(label, data):
        xlim = np.nanmax(data.values) if xmax is None else xmax
        plt.xlabel('Concentration (μg/L)', fontsize=12)
        plt.ylabel('Percentile', fontsize=12)
        plt.ylim([0, 101])
        plt.xlim([0, xlim])
        axis = plt.gca()
        axis.set_label(label)

    def write(f, clear=True):
        if combined or labels is not None or selection_label is not None:
            plt.legend(loc='lower right', title=combined_label)
        plt.savefig(plot_outfile.format(f), dpi=600)
        if clear:
            plt.clf()

    def overlay_selected():
        sel_concentrations, sel_percentiles, sel_targets = \
            selection.loc[selection.duration == field, ['concentration', 'percentile', 'target']].values.T
        for x, y in zip(sel_concentrations, sel_targets):
            plt.axhline(y=y, ls="--", lw=1)
            plt.axvline(x=x, ls="--", lw=1)
        plt.scatter(sel_concentrations, sel_percentiles, s=100, label=selection_label)

    outfiles = []
    if combined:
        initialize(combined_label, scenarios[fields])
    for i, field in enumerate(fields):
        concentrations, percentiles = scenarios[[field, percentile_field.format(field)]].values.T
        if combined:
            label = labels[i] if labels is not None else None
            plt.scatter(concentrations, percentiles, s=1, label=label)
            if selection is not None:
                overlay_selected()
        else:
            initialize(individual_label, scenarios[[field]])
            plt.scatter(concentrations, percentiles, s=1, label=labels)
            if selection is not None:
                overlay_selected()
            if plot_outfile is not None:
                write(field, True)
            outfiles.append(plot_outfile.format(field))
    if combined and plot_outfile is not None:
        write("combined")
    return outfiles


def process_single(scenarios=None, pwc_infile=None, pwc_outfile=None, region=None, crop=None, koc=None):
    if not all((region, crop, koc)):
        region, crop_id, crop, koc = detect_params(pwc_outfile)

    if scenarios is None:
        if all((pwc_infile, pwc_outfile)):
            scenarios = read_infiles(pwc_infile, pwc_outfile)
        else:
            raise ValueError("Scenario and PWC outfiles not specified")

    summary_outfile, selected_scenario_outfile, plot_outfile = \
        initialize_output(region, crop, koc)

    # Calculate percentiles for test fields and append additional attributes
    scenarios = compute_percentiles(scenarios, pwc_durations, area_weighting)
    scenarios.to_csv(summary_outfile, index=None)

    # Select scenarios for each duration based on percentile, and write to file
    selection = select_scenarios(scenarios, selection_percentiles, selection_window)
    selection.to_csv(selected_scenario_outfile, index=None)
    selection = streamline_selection(selection)

    # Plot results and save tabular data to file
    outfiles = plot(scenarios, pwc_durations, plot_outfile, selection, labels='Scenarios', selection_label='Selected',
                    combined=False)
    outfiles += plot(scenarios, pwc_durations, plot_outfile, selection, labels=pwc_durations)

    return [summary_outfile, selected_scenario_outfile] + outfiles


def process_singles(full_table):
    all_combinations = \
        full_table[['cdl_name', 'cdl', 'region', 'koc']].drop_duplicates().sort_values(['cdl', 'region', 'koc']).values

    durations = full_table[[scenario_id_field, 'duration', 'koc', 'conc', 'area']]
    ## TODO - there are scenarios with duplicate values... why??
    durations = durations.groupby([scenario_id_field, 'duration']).mean().reset_index()
    durations = durations.pivot(index=scenario_id_field, columns='duration', values='conc')
    scenarios = full_table.merge(durations, on=scenario_id_field, how='left')
    # TODO - there are rows with nan values... why?
    scenarios = scenarios.dropna()

    for cdl_name, cdl, region, koc in all_combinations:
        print(f"Working on Region {region} {cdl_name}, Koc {koc}")
        local_scenarios = scenarios[(scenarios.cdl == cdl) & (scenarios.region == region) & (scenarios.koc == koc)]
        try:
            process_single(local_scenarios, region=region, crop=cdl_name, koc=koc)
        except Exception as e:
            print(e)

def process_batch():
    # Find and categorize all input files
    file_map = fetch_files()

    # Read all tables into a single table
    full_table = combine_tables(file_map)

    # Plot national data by Koc and duration
    # Get output path for plots
    national_analysis(full_table, 'koc', 'duration')
    national_analysis(full_table, 'duration', 'koc')

    # Break down data by region/crop/koc
    process_singles(full_table)


process_batch()
