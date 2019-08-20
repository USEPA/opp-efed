"""
Searches for all PWC daily output files (chemographs) in a directory,
applies formatting, performs calculations, and writes to file
"""

import os
import re
import pandas as pd
import datetime as dt
import numpy as np


def find_files(file_dir, file_format, run_filter=None, scenario_filter=None):
    for f in os.listdir(file_dir):
        match = re.match(file_format, f)
        if match:
            run_id, scenario_id = match.groups()
            if (scenario_filter is None or scenario_id in scenario_filter) and \
                    (run_filter is None or run_id in run_filter):
                yield scenario_id, os.path.join(file_dir, f)


def daily_ranges(table, field):
    # Get daily values for monthly standard deviation, and use to create low/high daily values
    monthly_stdev = table[field].resample('M').transform('std')
    return table[field] - monthly_stdev, table[field] + monthly_stdev


def read_file(path, header):
    table = pd.read_csv(path, skiprows=5, names=header)
    with open(path) as f:
        line = [next(f) for _ in range(5)][-1]
    days_since_1900 = int(re.search("(\d+?) = Start Day", line).group(1))
    start_date = dt.datetime(1900, 1, 1) + dt.timedelta(days=days_since_1900)
    table['date'] = pd.date_range(start_date, periods=table.shape[0])
    table[['wc_conc', 'benthic_conc', 'peak_wc_conc']] *= 1000000.  # kg/m3 -> ug/L
    return table.set_index('date')


def write_to_file(table, outfile, header):
    pd.DataFrame(table, columns=header).to_csv(outfile, index=None)


def main():
    out_file = "Output/test_table.csv"
    time_series_dir = "J:/PWC/777e"
    file_format = re.compile("(.+?)_(.+?)_Pond_Parent_daily.csv")
    infile_header = ['depth', 'wc_conc', 'benthic_conc', 'peak_wc_conc']
    outfile_header = ['scenario_id', 'auc_sum', 'auc_trapezoid', 'auc_high', 'auc_low']

    # Initialize output table
    summary_table = []

    # Loop through all files in directory, apply filter if needed
    for scenario_id, file in find_files(time_series_dir, file_format):
        # Read the input file
        table = read_file(file, infile_header)

        # Demo of grouping by month
        table['wc_conc_low'], table['wc_conc_high'] = daily_ranges(table, 'wc_conc')

        # Calculate area under curve
        auc_trapezoid = np.trapz(table['wc_conc'])
        auc_sum = table['wc_conc'].sum()
        auc_high = table['wc_conc_high'].sum()
        auc_low = table['wc_conc_low'].sum()

        # Add summary results to output table
        summary_table.append([scenario_id, auc_sum, auc_trapezoid, auc_high, auc_low])

    print(summary_table)
    exit()
    # Write the summary table to output file
    write_to_file(summary_table, out_file, outfile_header)


main()
