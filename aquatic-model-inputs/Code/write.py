import os
import numpy as np
from paths import sam_scenario_path, pwc_scenario_path, recipe_path


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


def qc_table(qc_table, region):
    qc_table.to_csv(pwc_scenario_path.format(region, 'qc'))
