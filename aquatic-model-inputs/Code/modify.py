import numpy as np
import pandas as pd
from scipy import stats

import write
from utilities import fields, report
from parameters import max_horizons, hydro_soil_groups, uslep_values, aggregation_bins, depth_bins, hsg_cultivated, \
    hsg_non_cultivated, null_curve_number, usle_m_vals, usle_m_bins


def aggregate_soils(in_soils):
    from parameters import aggregation_bins

    # Sort data into bins
    out_data = [in_soils.hsg_letter, in_soils.state]
    for field, field_bins in aggregation_bins.items():
        # Designate aggregated field labels (e.g., sl1, sl2 for slope) and apply with 'cut'
        labels = [field[:2 if field == "slope" else 1] + str(i) for i in range(1, len(field_bins))]
        sliced = pd.cut(in_soils[field].fillna(0), field_bins, labels=labels, right=False, include_lowest=True)
        out_data.append(sliced.astype("str"))
    soil_agg = pd.concat(out_data, axis=1)

    # Create aggregation key
    invalid = pd.isnull(soil_agg[['state', 'hsg_letter', 'slope', 'orgC_5', 'sand_5', 'clay_5']]).any(axis=1)
    in_soils.loc[:, 'soil_id'] = 'null'
    in_soils.loc[~invalid, 'soil_id'] = soil_agg['state'] + soil_agg['hsg_letter'] + soil_agg['slope'] + soil_agg[
        'orgC_5'] + soil_agg['sand_5'] + soil_agg['clay_5']

    aggregation_key = in_soils[['mukey', 'soil_id', 'state']]

    del in_soils['mukey']

    # Group by aggregation key and take the mean of all properties except HSG, which will use mode
    grouped = in_soils.groupby('soil_id')
    averaged = grouped.mean().reset_index()
    hydro_group = grouped['hydro_group'].agg(lambda x: stats.mode(x)[0][0]).to_frame().reset_index()
    del averaged['hydro_group']
    in_soils = averaged.merge(hydro_group, on='soil_id')

    return in_soils, aggregation_key


def crop_data(crop_table):
    # Use active dates
    crop_table['plant_begin'] = crop_table['plant_begin_active']
    crop_table['plant_end'] = crop_table['plant_end_active']
    crop_table['harvest_begin'] = crop_table['harvest_begin_active']
    crop_table['harvest_end'] = crop_table['harvest_end_active']
    return crop_table


def depth_weight_soils(in_soils):
    # Get the root name of depth weighted fields
    depth_fields = fields.fetch('depth_weight')

    # Generate weighted columns for each bin
    depth_weighted = []
    for bin_top, bin_bottom in zip([0] + list(depth_bins[:-1]), list(depth_bins)):
        bin_table = np.zeros((in_soils.shape[0], len(depth_fields)))

        # Perform depth weighting on each horizon
        for i in range(max_horizons):
            # Set field names for horizon
            top_field = 'horizon_top_{}'.format(i + 1)
            bottom_field = 'horizon_bottom_{}'.format(i + 1)
            value_fields = ["{}_{}".format(f, i + 1) for f in depth_fields]

            # Adjust values by bin
            horizon_bottom, horizon_top = in_soils[bottom_field], in_soils[top_field]
            overlap = (horizon_bottom.clip(upper=bin_bottom) - horizon_top.clip(lower=bin_top)).clip(0)
            ratio = (overlap / (horizon_bottom - horizon_top)).fillna(0)
            bin_table += in_soils[value_fields].fillna(0).mul(ratio, axis=0).values

        # Add columns
        bin_table = \
            pd.DataFrame(bin_table, columns=["{}_{}".format(f, bin_bottom) for f in depth_fields])
        depth_weighted.append(bin_table)

    # Clear all fields corresponding to horizons, and add depth-binned data
    fields.expand('horizon')  # this will add all the _n fields
    for field in fields.fetch('horizontal'):
        del in_soils[field]
    in_soils = pd.concat([in_soils.reset_index()] + depth_weighted, axis=1)

    return in_soils


def combinations(combos, crop_data, soil_data, met_data, exclude, aggregation_key):
    # Split double-cropped classes into individual scenarios
    double_crops = crop_data[['cdl', 'cdl_alias']].drop_duplicates().sort_values('cdl')
    combos = combos.merge(double_crops, on='cdl', how='left')

    # Modify soil id for aggregated scenarios
    if aggregation_key is not None:
        combos = combos.merge(aggregation_key, on='mukey', how="left")
        del combos['mukey']
        id_columns = [c for c in combos.columns if c != "area"]
        combos = combos.groupby(id_columns).sum().reset_index()
    else:
        combos = combos.rename(columns={'mukey': 'soil_id'})

    # Scenarios are only selected for PWC where data is complete
    if exclude:
        combos = combos.merge(soil_data[['soil_id', 'state']], on='soil_id', how='inner')
        available_crops = crop_data.dropna(subset=fields.fetch('CropDates'), how='all')[['cdl', 'cdl_alias', 'state']]
        combos = combos.merge(available_crops, on=['cdl', 'cdl_alias', 'state'], how='inner')
    else:
        # Add a 'state' parameter to the combinations from soil map unit link
        combos = combos.merge(soil_data[['soil_id', 'state']], on='soil_id', how='left')

    # Create a unique identifier
    combos['scenario_id'] = combos.state + \
                            'S' + combos.soil_id.astype("str") + \
                            'W' + combos.weather_grid.astype("str") + \
                            'LC' + combos.cdl.astype("str")
    return combos


def soils(in_soils, depth_weight=False, aggregate=False):
    from parameters import o_horizon_max, slope_length_max, slope_min

    """  Identify component to be used for each map unit """
    fields.refresh()

    # Adjust soil data values
    in_soils.loc[:, 'orgC'] /= 1.724  # oc -> om
    in_soils.loc[:, ['fc', 'wp']] /= 100.  # pct -> decimal
    in_soils.loc[pd.isnull(in_soils.slope_length), 'slope_length'] = slope_length_max
    in_soils.loc[in_soils.slope < slope_min, 'slope'] = slope_min

    # Isolate unique map unit/component pairs and select major component with largest area (comppct)
    components = in_soils[['mukey', 'cokey', 'major_component', 'component_pct']].drop_duplicates(['mukey', 'cokey'])
    components = components[components.major_component == 'Yes']
    components = components.sort_values('component_pct', ascending=False)
    components = components[~components.mukey.duplicated()]
    in_soils = components[['mukey', 'cokey']].merge(in_soils, on=['mukey', 'cokey'], how='left')

    # Delete thin organic horizons
    in_soils = in_soils[~((in_soils.horizon_letter == 'O') &
                          (in_soils.horizon_bottom <= o_horizon_max))]

    # Sort table by horizon depth and get horizon information
    in_soils = in_soils.sort_values(['cokey', 'horizon_top'])
    in_soils['thickness'] = in_soils['horizon_bottom'] - in_soils['horizon_top']
    in_soils['horizon_num'] = in_soils.groupby('cokey').cumcount() + 1
    in_soils = in_soils.sort_values('horizon_num', ascending=False)
    in_soils[~in_soils.horizon_num > max_horizons]

    # Extend columns of data for multiple horizons
    horizon_data = in_soils.set_index(['cokey', 'horizon_num'])[fields.fetch('horizontal')]
    horizon_data = horizon_data.unstack().sort_index(1, level=1)
    horizon_data.columns = ['_'.join(map(str, i)) for i in horizon_data.columns]
    for f in fields.fetch('horizontal'):  # Create empty fields up to n_horizons (pwc horizons are fixed)
        for i in range(in_soils.horizon_num.max(), max_horizons + 1):
            horizon_data["{}_{}".format(f, i)] = np.nan
        del in_soils[f]
    in_soils = in_soils.drop_duplicates(['mukey', 'cokey']).merge(horizon_data, left_on='cokey', right_index=True)
    in_soils = in_soils.rename(columns={'horizon_num': 'n_horizons'})

    # New HSG code - take 'max' of two versions of hsg
    hsg_to_num = {hsg: i + 1 for i, hsg in enumerate(hydro_soil_groups)}
    num_to_hsg = {v: k.replace("/", "") for k, v in hsg_to_num.items()}
    in_soils['hydro_group'] = in_soils[['hydro_group', 'hydro_group_dominant']].applymap(
        lambda x: hsg_to_num.get(x)).max(axis=1).fillna(-1).astype(np.int32)
    in_soils['hsg_letter'] = in_soils['hydro_group'].map(num_to_hsg)

    # Select USLE variables
    # Take the value from the top horizon with valid kwfact values
    in_soils['usle_k'] = in_soils[["usle_k_{}".format(i + 1) for i in range(max_horizons)]].bfill(1).iloc[:, 0]
    m = usle_m_vals[np.int16(pd.cut(in_soils.slope.values, usle_m_bins, labels=False))]
    sine_theta = np.sin(np.arctan(in_soils.slope / 100))  # % -> sin(rad)
    in_soils['usle_ls'] = (in_soils.slope_length / 72.6) ** m * (65.41 * sine_theta ** 2. + 4.56 * sine_theta + 0.065)
    in_soils['usle_p'] = np.array(uslep_values)[
        np.int16(pd.cut(in_soils.slope, aggregation_bins['slope'], labels=False))]

    # Set n_horizons to the first invalid horizon
    horizon_fields = [f for f in fields.fetch('horizontal') if f in fields.fetch('pwc_scenario')]
    fields.expand('horizon')
    qc_table = fields.perform_qc(in_soils).copy()
    for field in horizon_fields:
        check_fields = ['{}_{}'.format(field, i + 1) for i in range(max_horizons)]
        if qc_table[check_fields].values.max() > 1:  # QC value of 2 indicates invalid data
            violations = (qc_table[check_fields] > 2).values
            keep_horizons = np.where(violations.any(1), violations.argmax(1), max_horizons)
            in_soils['n_horizons'] = np.minimum(in_soils.n_horizons.values, keep_horizons)
    in_soils['n_horizons'] = in_soils.n_horizons.fillna(0).astype(np.int32)

    # Adjust cumulative thickness
    profile = in_soils[['thickness_{}'.format(i + 1) for i in range(max_horizons)]]
    profile_depth = profile.mask(~np.greater.outer(in_soils.n_horizons, np.arange(max_horizons))).sum(axis=1)
    in_soils['root_zone_max'] = np.minimum(in_soils.root_zone_max.values, profile_depth)

    if depth_weight:
        in_soils = depth_weight_soils(in_soils)
    if aggregate:
        in_soils, aggregation_key = aggregate_soils(in_soils)
    else:
        in_soils = in_soils.rename(columns={'mukey': 'soil_id'})
        aggregation_key = None

    return in_soils, aggregation_key


def scenarios(in_scenarios, exclude, depth_weight, mode, region, perform_qc=True):
    in_scenarios['emergence_begin'] = np.int32(in_scenarios.plant_begin + 7)
    in_scenarios['emergence_end'] = np.int32(in_scenarios.plant_end + 7)
    in_scenarios['maxcover_begin'] = np.int32((in_scenarios.plant_begin + in_scenarios.harvest_begin) / 2)
    in_scenarios['maxcover_end'] = np.int32((in_scenarios.plant_end + in_scenarios.harvest_end) / 2)

    # Select crop dates for PWC runs
    # TODO - what are we doing for sam exactly?
    if mode == 'pwc':
        for stage in ('plant', 'emergence', 'maxcover', 'harvest'):
            new_field = '{}_date'.format(stage)
            in_scenarios[new_field] = \
                (in_scenarios['{}_begin'.format(stage)] + in_scenarios['{}_end'.format(stage)]) / 2
            in_scenarios.loc[pd.isnull(in_scenarios[new_field]), new_field] = 0
            in_scenarios[new_field] = in_scenarios[new_field].astype(np.int16)

    # Select curve numbers
    in_scenarios['cn_cov'] = null_curve_number
    in_scenarios['cn_fal'] = null_curve_number
    for cultivated, soil_groups in enumerate((hsg_non_cultivated, hsg_cultivated)):
        for group_num, group_name in enumerate(soil_groups):
            sel = (in_scenarios.hydro_group == group_num + 1) & (in_scenarios.cultivated_cdl == cultivated)
            for var in 'cov', 'fal':
                in_scenarios.loc[sel, 'cn_' + var] = in_scenarios.loc[sel, 'cn_{}_{}'.format(var, group_name)]

    # Ensure that root and evaporation depths are 0.5 cm or more shallower than soil depth
    in_scenarios['root_depth'] = \
        np.minimum(in_scenarios.root_zone_max.values - 0.5, in_scenarios.amxdr.values)
    in_scenarios['evaporation_depth'] = \
        np.minimum(in_scenarios.root_zone_max.values - 0.5, in_scenarios.anetd)

    # Recode for old met station ids (6/29 - overlay performed with old grid, save for future)
    # scenarios['weather_grid'] = scenarios['stationID']

    # Select output fields
    fields.refresh()
    other_fields = [f for f in fields.fetch('pwc_scenario') if not f in fields.fetch('horizontal')]
    fields.expand('depth') if depth_weight else fields.expand('horizon')
    if exclude:
        qc_table = fields.perform_qc(in_scenarios).copy()
        if perform_qc:
            scenarios_qc(in_scenarios, qc_table, other_fields, region, mode)
        invalid_rows = qc_table[other_fields].max(axis=1) == 2
        before_area = in_scenarios.area.sum()
        in_scenarios = in_scenarios[~invalid_rows]
        ratio = int((1 - (in_scenarios.area.sum() / before_area)) * 100.)
        report("Removing {}% of scenario area due to invalid values".format(ratio))
    return in_scenarios[fields.fetch(mode + '_scenario')]


def scenarios_qc(in_scenarios, qc_table, other_fields, region, mode):
    violation_table = in_scenarios[
        ['scenario_id', 'state', 'soil_id', 'weather_grid', 'cdl', 'cdl_alias', 'muname']]
    field_report = [['n scenarios', in_scenarios.shape[0]]]
    for field in other_fields:
        bogeys = np.where(qc_table[field] > 2)[0]
        if bogeys.sum() > 0:
            field_report.append([field, (bogeys > 0).sum()])
            new_field = 'violation_{}'.format(field)
            violation_table[new_field] = 0
            violation_table[field] = in_scenarios[field]
            violation_table.loc[bogeys, new_field] = 1
    violation_fields = [f for f in violation_table.columns if "violation" in f]
    violation_table['n_violations'] = violation_table[violation_fields].sum(axis=1)
    field_report = pd.DataFrame(field_report, columns=['field', 'n_violations'])
    write.scenarios(violation_table, mode, region, 'qc')
    write.scenarios(field_report, mode, region, 'report')


if __name__ == "__main__":
    __import__('1_scenarios_and_recipes').main()
