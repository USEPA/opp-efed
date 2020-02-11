import os

import pandas as pd

from parameters import states_nhd, vpus_nhd
from paths import condensed_soil_path, nhd_path, met_attributes_path, combo_path, crop_dates_path, \
    crop_params_path, gen_params_path, irrigation_path
from utilities import fields, read_dbf, report


def combinations(region, year, mode):
    if mode == 'sam':
        out_header = ['gridcode', 'cdl', 'weather_grid', 'mukey', 'area']
    else:
        out_header = ['cdl', 'weather_grid', 'mukey', 'area']
    return pd.read_csv(combo_path.format(region, year))[out_header]


def crop():
    crop_params = pd.read_csv(crop_params_path)
    crop_dates = pd.read_csv(crop_dates_path)
    irrigation_data = pd.read_csv(irrigation_path).rename(columns={'cdl': 'cdl_alias'})
    genclass_params = pd.read_csv(gen_params_path)

    # Merge all crop-related data tables
    data = crop_params \
        .merge(crop_dates, on=['cdl', 'cdl_alias'], how='left', suffixes=('', '_burn')) \
        .merge(irrigation_data, on=['cdl_alias', 'state'], how='left') \
        .merge(genclass_params, on='gen_class', how='left', suffixes=('_cdl', '_gen'))
    data = data[[c for c in data.columns if not c.endswith("_burn")]]

    # Convert to dates
    for field in fields.fetch('CropDates'):
        data[field] = (pd.to_datetime(data[field], format="%d-%b") - pd.to_datetime("1900-01-01")).dt.days

    # Where harvest is before plant, add 365 days (e.g. winter wheat)
    data.loc[data.plant_begin > data.harvest_begin, 'harvest_begin'] += 365
    data.loc[data.plant_end > data.harvest_end, 'harvest_end'] += 365
    data.loc[data.plant_begin_active > data.harvest_begin_active, 'harvest_begin_active'] += 365
    data.loc[data.plant_end_active > data.harvest_end_active, 'harvest_end_active'] += 365

    return data


def nhd_params(region):
    gridcodes_path = \
        os.path.join(nhd_path.format(vpus_nhd[region], region), "NHDPlusCatchment", "featureidgridcode.dbf")
    return read_dbf(gridcodes_path)[['featureid', 'gridcode']].rename(columns={"featureid": "comid"})


def met():
    met_data = pd.read_csv(met_attributes_path)
    del met_data['weather_grid']
    met_data = met_data.rename(columns={"stationID": 'weather_grid'})  # these combos have old weather grids?
    return met_data


def ssurgo(s, name):
    table_path = condensed_soil_path.format(s, name)
    table_fields = fields.fetch(name, 'external')
    data_types = fields.data_type(fetch=name, old_fields=True)
    return pd.read_csv(table_path, dtype=data_types)[table_fields]


def soils(region=None, state=None):
    fields.refresh()

    if region is None and state is not None:
        region_states = [state]
    else:
        region_states = states_nhd[region]
    state_tables = []
    valu_table = ssurgo("", "valu")
    for state in region_states:
        state_table = None
        for table_name, key_field in [('muaggatt', 'mukey'), ('component', 'mukey'), ('chorizon', 'cokey')]:
            table = ssurgo(state, table_name)
            state_table = table if state_table is None else pd.merge(state_table, table, on=key_field, how='outer')
        state_table['state'] = state
        state_tables.append(state_table)
    soil_data = pd.concat(state_tables, axis=0)
    soil_data = soil_data.merge(valu_table, on='mukey')
    return soil_data.rename(columns=fields.convert)


if __name__ == "__main__":
    __import__('1_scenarios_and_recipes').main()
