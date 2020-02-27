import os
import pandas as pd
import numpy as np
from paths import sam_scenario_path, pwc_scenario_path, recipe_path
from utilities import fields


def create_dir(out_path):
    """ Create a directory for a file name if it doesn't exist """
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))


def scenarios(scenario_matrix, mode, region, name=None):
    if scenario_matrix is not None:
        if mode == 'sam':
            name = "" if name is None else name
            out_path = sam_scenario_path.format(region + name)
        elif mode == 'pwc':
            out_path = pwc_scenario_path.format(region, name)
        out_path = out_path.replace("/", "-")
        create_dir(out_path)
        scenario_matrix.to_csv(out_path, index=False)


def recipes(region, year, recipe_matrix, recipe_map):
    np.savez_compressed(recipe_path.format(region, year), recipes=recipe_matrix, map=recipe_map)


def qc_report(in_scenarios, qc_table, qc_fields, region, mode):
    # Initialize a table with all violations for each scenario
    violation_table = in_scenarios[[f for f in fields.fetch('id') if f in in_scenarios.columns]]

    # Initialize a report with violations by field
    field_report = [['n scenarios', in_scenarios.shape[0]]]

    # Iterate through fields and test for violations
    for field in qc_table.columns:
        violation_table[field] = 0
        bogeys = np.where(qc_table[field] > 2)[0]
        if bogeys.sum() > 0:
            violation_table.loc[bogeys, field] = 1
            field_report.append([field, (bogeys > 0).sum()])

    # Report on the violations for each scenario
    violation_table['n_violations'] = violation_table[qc_fields].sum(axis=1)
    scenarios(violation_table, mode, region, 'qc')

    # Report on the number of violations for each field
    field_report = pd.DataFrame(field_report, columns=['field', 'n_violations'])
    scenarios(field_report, mode, region, 'report')
