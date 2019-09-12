import os
import pandas as pd

pwc_run_id = ['run_id', 'pwc_scenario_id', 'rep']
pwc_id_fields = ['line_num', 'run_id']
pwc_durations = ['Peak', '1-day', 'Yr', 'overall', '4-day', '21-day', '60-day', '90-day', 'PW_pk', 'PW_21']
pwc_header = pwc_id_fields + pwc_durations
new_cols = dict(zip(['1-day', 'Yr', 'overall'], ['acute', 'chronic', 'cancer']))


# acute - 1 day, chronic - 1 year, cancer, overall

def read_pwc_output(in_file):
    # Read the table, manually entering in the header (original header is tough to parse)
    table = pd.read_csv(in_file, names=['bull'] + pwc_header, delimiter=r'\s+', skiprows=1)

    # Adjust line number so that header is not included
    table['line_num'] -= 1

    # Split the Batch Run ID field into constituent parts
    data = table.pop('run_id').str.split('_', expand=True)
    data.columns = pwc_run_id

    return pd.concat([data, table], axis=1)


root_dir = r"C:\Users\Jhook\Environmental Protection Agency (EPA)\Spatial Aquatic Model (SAM) Development - Documents\PWC Scenarios\PFTTT_ScenariosProject\ScenarioBatchFiles"
pwc_output_dir = os.path.join(root_dir, "Corn", "Comprehensive_Batch_Summary Files")
scenario_dir = os.path.join(root_dir, "Corn", "PWC_Field_Scenarios_Corn")

for f in os.listdir(pwc_output_dir):
    if 'r03N_1_corn_koc10000' in f:
        print(f)
        path = os.path.join(pwc_output_dir, f)
        output = pd.read_csv(path, sep="\s+", skiprows=1, names=pwc_header)[['line_num', 'run_id', '1-day', 'Yr', 'overall']]
        output.to_csv(path, index=None, header=False, sep="\t")
        #output = pd.read_csv(path, names=['a', 'b', 'c', 'd', 'e'])
        #output.to_csv(path, sep="\t", index=None, header=False)

