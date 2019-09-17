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
    full_table[scenario_id_field] = full_table[scenario_id_field].astype(object)
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
        outfiles = ['{}_{}_{}']  # type, duration, koc
    results_dir = os.path.join(output_dir, tag + "summary_files")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return [os.path.join(results_dir, tag + t) for t in outfiles]


def pivot_scenarios(scenarios, mode):
    if mode == 'koc':
        field1, field2 = 'koc', 'duration'
    elif mode == 'duration':
        field1, field2 = 'duration', 'koc'
    else:
        raise ValueError(f"Invalid mode '{mode}'")
    root_table = scenarios.drop(columns=[field1, 'conc']).drop_duplicates()
    local = scenarios[[scenario_id_field, field1, 'conc']]
    # TODO - there are scenarios with duplicate values... why??
    local = local.groupby([scenario_id_field, field1]).mean().reset_index()
    local = local.pivot(index=scenario_id_field, columns=field1, values='conc').reset_index()
    local = root_table.merge(local, on=scenario_id_field, how='left')
    return local.dropna()  # TODO - there are rows with nan values... why?


def national_analysis(table, field1, field2):
    # TODO - cdl class?
    plot_outfile = initialize_output()[0]
    table = table[[scenario_id_field, 'duration', 'koc', 'conc', 'area']]
    fields = sorted(table[field2].unique())
    for selection_val, subset in table.groupby(field1):
        subset = pivot_scenarios(subset, field2)
        subset = compute_percentiles(subset, fields)
        plot(subset, fields, combined_label=field2.capitalize(), labels=fields,
             plot_outfile=plot_outfile.format('concs', selection_val, field2), xmax=120)


def read_infiles(pwc_infile, pwc_outfile):
    # Join the scenarios data with the computed concentrations
    errors = []
    infiles = []
    for infile, read_function in ((pwc_infile, read_scenarios), (pwc_outfile, read_pwc_output)):
        try:
            table = read_function(infile)
            infiles.append(table)
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


def select_scenarios(scenarios, fields, method='window'):
    # Designate the lower and upper bounds for the percentile selection
    window = selection_window / 2  # half below, half above
    # Select scenarios for each of the durations and combine
    all_selected = []
    for conc_field, selection_pct in itertools.product(fields, selection_percentiles):
        pct_field = percentile_field.format(conc_field)
        if method == 'window':
            # Selects all scenarios within the window, or outside but with equal value
            selected = scenarios[(scenarios[pct_field] >= (selection_pct - window)) &
                                 (scenarios[pct_field] <= (selection_pct + window))]
            min_val, max_val = selected[pct_field].min(), selected[pct_field].max()
            selection = scenarios[(scenarios[pct_field] >= min_val) & (scenarios[pct_field] <= max_val)]
        elif method == 'nearest':
            rank = (scenarios[pct_field] - selection_pct).abs().sort_values().index
            selection = scenarios.loc[rank].iloc[0].to_frame().T

        # Set new fields
        rename = {conc_field: 'concentration', pct_field: 'percentile'}
        selection = selection[scenario_fields + list(rename.keys())].rename(columns=rename)
        for col in ['concentration', 'percentile']:
            selection[col] = selection[col].astype(np.float32)
        selection['target'] = selection_pct
        selection['subject'] = conc_field
        all_selected.append(selection)

    all_selected = \
        pd.concat(all_selected, axis=0).sort_values(['subject', 'area'], ascending=[True, False]).reset_index()

    return all_selected


def streamline_selection(selection):
    """ Ranges of selections don't plot well. Take the ones with the largest areas for plotting """
    selection = selection.reset_index()
    selection = selection.loc[selection.groupby(['target', 'subject'])['area'].idxmax()]
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

    def overlay_selected(label=True):
        sel_concentrations, sel_percentiles, sel_targets = \
            selection.loc[selection.subject == field, ['concentration', 'percentile', 'target']].values.T
        for x, y in zip(sel_concentrations, sel_targets):
            plt.axhline(y=y, ls="--", lw=1)
            plt.axvline(x=x, ls="--", lw=1)
            if label:
                plt.text(1.5, y, round(y, 1))
                plt.text(x, 1.5, round(x, 1))
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
                overlay_selected(False)
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


def report_single(scenarios, region, crop, koc):
    summary_outfile, selected_scenario_outfile, plot_outfile = \
        initialize_output(region, crop, koc)

    # Calculate percentiles for test fields and append additional attributes
    scenarios = compute_percentiles(scenarios, pwc_durations, area_weighting)

    # Select scenarios for each duration based on percentile, and write to file
    selection = select_scenarios(scenarios, pwc_durations, method='window')

    # Write to files
    scenarios.to_csv(summary_outfile, index=None)
    selection.to_csv(selected_scenario_outfile, index=None)

    # Plot results and save tabular data to file
    selection = streamline_selection(selection)
    outfiles = plot(scenarios, pwc_durations, plot_outfile, selection, labels='Scenarios', selection_label='Percentile',
                    combined=False)
    outfiles += plot(scenarios, pwc_durations, plot_outfile, selection, labels=pwc_durations)

    return [summary_outfile, selected_scenario_outfile] + outfiles


def batch_singles(scenarios):
    for (cdl_name, cdl, region, koc), local_scenarios in scenarios.groupby(['cdl_name', 'cdl', 'region', 'koc']):
        print(f"Working on {region} {koc}...")
        local_scenarios = pivot_scenarios(local_scenarios, 'duration')
        report_single(local_scenarios, region=region, crop=cdl_name, koc=koc)


def get_ratios(full_table):
    out_path = initialize_output()[0]
    regions = sorted(full_table.region.unique())
    all_tables = []
    for region, koc in itertools.product(regions, kocs):
        print(region)
        regional_table = full_table[(full_table.region == region) & (full_table.koc == str(koc))]
        print(regional_table.empty)
        if regional_table.empty:
            print("yea no")
            continue
        regional_table = pivot_scenarios(regional_table, 'duration')
        regional_table = compute_percentiles(regional_table, pwc_durations)
        selection = select_scenarios(regional_table, pwc_durations, method='nearest')
        selection = selection.groupby(['subject', 'target']).mean().reset_index()
        selection = selection.pivot(index='subject', columns='target', values='concentration')
        for pct in map(int, selection_percentiles[1:]):
            try:
                selection[f'r_{pct}/50'] = selection[pct] / selection[50]
            except KeyError:
                print(f"No scenarios found for {region} {koc}")
        selection['region'] = region
        selection['koc'] = koc
        all_tables.append(selection)
    ratio_table = pd.concat(all_tables, axis=0).reset_index()
    ratio_table.to_csv(out_path.format('ratios', 'all', '_all') + ".csv", index=None)

    # Plot output
    regions = [''] + sorted(ratio_table.region.unique())
    region_labels = [r.lstrip("r") for r in regions]
    for koc, duration in itertools.product(kocs, pwc_durations):
        table = ratio_table[(ratio_table.subject == duration) & (ratio_table.koc == koc)]
        plt.xlabel('Region', fontsize=12)
        plt.ylabel('Concentration (μg/L)', fontsize=12)
        plt.xticks(range(len(regions)), region_labels, size='small')
        plt.xlim([0, 22])
        region_index = np.array([regions.index(r) for r in table.region])
        for pct in selection_percentiles:
            plt.scatter(region_index, table[pct], s=10, label=pct)
        plt.legend(loc='best', title='Percentiles')
        plt.savefig(out_path.format('ratios', duration, koc), dpi=600)
        plt.clf()


def main():
    import time
    start = time.time()
    # Find and categorize all input files
    file_map = fetch_files()

    # Read all tables into a single table
    full_table = combine_tables(file_map)

    # Get ratios
    get_ratios(full_table)

    # Plot national data by Koc and duration
    national_analysis(full_table, 'koc', 'duration')
    national_analysis(full_table, 'duration', 'koc')
    # exit()
    # Break down data by region/crop/koc
    batch_singles(full_table)
    print(time.time() - start)

if __name__ == "__main__":
    main()
