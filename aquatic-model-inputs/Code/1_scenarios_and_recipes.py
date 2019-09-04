import numpy as np
import pandas as pd
import modify
import read
import write

from utilities import report
from parameters import nhd_regions

from parameters import pwc_selection_field as crop_field
from parameters import pwc_selection_pct as selection_pct
from parameters import pwc_min_selection as min_sample


def create_recipes(combinations, watershed_data):
    # Convert gridcode to comid
    combinations = combinations.merge(watershed_data, on='gridcode').sort_values('comid')
    comids = combinations.pop('comid').values
    recipes = combinations[['scenario_id', 'area']]

    # Get the first and last row corresponding to each comid
    first_rows = np.hstack(([0], np.where(comids[:-1] != comids[1:])[0] + 1))
    last_rows = np.hstack((first_rows[1:], [comids.size]))
    recipe_map = np.vstack((comids[first_rows], first_rows, last_rows)).T

    # Once recipes are created, remove watershed id from scenarios and collapse
    id_columns = [c for c in combinations if c != 'area']
    combinations = combinations.groupby(id_columns).sum().reset_index()

    return recipes, recipe_map, combinations


def create_scenarios(combinations, soil_data, crop_data, met_data):
    scenarios = combinations.merge(soil_data, how="left", on="soil_id", suffixes=("", "_soil"))
    scenarios = scenarios.merge(crop_data, how="left", on=['cdl', 'cdl_alias', 'state'])
    scenarios = scenarios.merge(met_data, how="left", on="weather_grid")
    return scenarios


def select_pwc_scenarios(in_scenarios, crop_data):
    # Randomly sample from each crop group and save the sample
    meta_table = []  # table summarizing sample size for each crop
    crop_groups = crop_data[[crop_field, crop_field + '_desc']].drop_duplicates().values

    # First, write the entire scenario table to a 'parent' table
    yield 'parent', in_scenarios

    # Write a table for each crop or crop group
    for crop, crop_name in crop_groups:
        sample = in_scenarios.loc[in_scenarios[crop_field] == crop]
        n_scenarios = sample.shape[0]
        selection_size = max((min_sample, int(n_scenarios * (selection_pct / 100))))
        if n_scenarios > selection_size:
            sample = sample.sample(selection_size)
        if not sample.empty:
            meta_table.append([crop, crop_name, n_scenarios, min((n_scenarios, selection_size))])
            yield '{}_{}'.format(crop, crop_name), sample

    # Write a table describing how many scenarios were selected for each crop
    yield 'meta', pd.DataFrame(np.array(meta_table), columns=['crop', 'crop_name', 'n_scenarios', 'sample_size'])


def update_combinations(all_combos, new_combos):
    del new_combos['gridcode']

    # Append new scenarios to running list
    all_combos = pd.concat([all_combos, new_combos], axis=0) if all_combos is not None else new_combos

    # Combine duplicate scenarios by adding areas
    all_combos = all_combos.groupby([c for c in all_combos.columns if not c == 'area']).sum().reset_index()

    return all_combos


def scenarios_and_recipes(regions, years, mode):
    # Read data indexed to weather grid
    met_data = read.met()

    # Read data indexed to crop
    crop_data = read.crop()
    crop_data = modify.land_use(crop_data, mode)

    # Soils, watersheds and combinations are broken up by NHD region
    for region in regions:
        report("Processing Region {}...".format(region))

        # Read data indexed to watershed
        watershed_data = read.nhd_params(region) if mode == 'sam' else None

        # Read data indexed to soil and process
        soil_data = read.soils(region)
        soil_data = modify.soils(soil_data, mode)

        # Initialize combinations for all years
        all_combinations = None
        for year in years:
            report("Reading combos for {}...".format(year), 1)

            # Read met/crop/land cover/soil/watershed combinations and process
            combinations = read.combinations(region, year)
            combinations = \
                modify.combinations(combinations, crop_data, soil_data, met_data, mode)

            # If running in SAM mode, create recipes and yearly scenarios
            if mode == 'sam':
                report("Building recipes...", 2)
                recipes, recipe_map, combinations = create_recipes(combinations, watershed_data)
                write.recipes(region, year, recipes, recipe_map)
            all_combinations = update_combinations(all_combinations, combinations)

        # Join combinations with data tables and perform additional attribution
        scenarios = create_scenarios(all_combinations, soil_data, crop_data, met_data)
        scenarios = modify.scenarios(scenarios, mode, region)

        # Write scenarios to file
        if mode == 'pwc':
            for crop_name, crop_scenarios in select_pwc_scenarios(scenarios, crop_data):
                report("Creating table for Region {} {}...".format(region, crop_name), 1)
                write.scenarios(crop_scenarios, mode, region, name=crop_name)
        else:
            write.scenarios(scenarios, mode, region)


def main():
    modes = ('sam',)  # pwc and/or sam
    years = range(2013, 2015)
    regions = nhd_regions

    for mode in modes:
        # Automatically adjust run parameters for pwc or sam
        scenarios_and_recipes(regions, years, mode)


if __name__ == "__main__":
    main()
