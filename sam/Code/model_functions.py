import dask
import numpy as np
from numba import guvectorize, njit


# TODO - comments back in
# TODO - iteration reductions?


@njit
def benthic_loop(eroded_soil, erosion_mass, soil_volume):
    benthic_mass = np.zeros(erosion_mass.size, dtype=np.float32)
    benthic_mass[0] = erosion_mass[0]
    for i in range(1, erosion_mass.size):
        influx_ratio = eroded_soil[i] / (eroded_soil[i] + soil_volume)
        benthic_mass[i] = (benthic_mass[i - 1] * (1. - influx_ratio)) + (erosion_mass[i] * (1. - influx_ratio))
    return benthic_mass


def compute_concentration(transported_mass, runoff, n_dates, q, debug=False):
    """
    Calculates pesticide concentration in water column from runoff inputs, accounting for time of travel
    Need to add references: VVWM (for basics), SAM write-up on time of travel
    """

    mean_runoff = runoff.mean()  # m3/d
    baseflow = np.subtract(q, mean_runoff, out=np.zeros(n_dates), where=(q > mean_runoff))
    total_flow = runoff + baseflow
    concentration = np.divide(transported_mass, total_flow, out=np.zeros(n_dates), where=(total_flow != 0))
    runoff_concentration = np.divide(transported_mass, runoff, out=np.zeros(n_dates), where=(runoff != 0))
    if debug:
        print(debug, "total_flow", total_flow.sum())
        print(debug, "average_conc", runoff_concentration.mean() * 1000000.)
    return total_flow, map(lambda x: x * 1000000., (concentration, runoff_concentration))  # kg/m3 -> ug/L


@guvectorize(['void(float64[:], int16[:], int16[:], int16[:], float64[:])'], '(p),(o),(o),(p)->(o)')
def exceedance_probability(time_series, window_sizes, thresholds, years_since_start, res):
    # Count the number of times the concentration exceeds the test threshold in each year
    n_years = years_since_start.max()
    for test_number in range(window_sizes.size):
        window_size = window_sizes[test_number]
        threshold = thresholds[test_number]
        if threshold == 0:
            res[test_number] = -1
        else:
            window_sum = np.sum(time_series[:window_size])
            exceedances = np.zeros(n_years)
            for day in range(window_size, len(time_series)):
                year = years_since_start[day]
                window_sum += time_series[day] - time_series[day - window_size]
                avg = window_sum / window_size
                if avg > threshold:
                    exceedances[year] = 1
            res[test_number] = exceedances.sum() / n_years


def field_to_soil(application_mass, rain, plant_factor, soil_2cm, foliar_degradation, washoff_coeff, covmax):
    # Initialize output
    pesticide_mass_soil = np.zeros(rain.size)
    canopy_mass, last_application = 0, 0  # Running variables

    # Determine if any pesticide has been applied to canopy
    canopy_applications = application_mass[1].sum() > 0

    # Loop through each day
    for day in range(plant_factor.size):

        # Start with pesticide applied directly to soil
        pesticide_mass_soil[day] = application_mass[0, day] * soil_2cm

        # If pesticide has been applied to the canopy, simulate movement from canopy to soil
        if canopy_applications:
            if application_mass[1, day] > 0:  # Pesticide applied to canopy on this day

                # Amount of pesticide intercepted by canopy
                canopy_pesticide_additions = application_mass[1, day] * plant_factor[day] * covmax

                # Add non-intercepted pesticide to soil
                pesticide_mass_soil[day] += (application_mass[1, day] - canopy_pesticide_additions) * soil_2cm
                canopy_mass = canopy_pesticide_additions + \
                              canopy_mass * np.exp((day - last_application) * foliar_degradation)
                last_application = day

            if rain[day] > 0:  # Simulate washoff
                canopy_mass *= np.exp((day - last_application) * foliar_degradation)
                pesticide_remaining = max(0, canopy_mass * np.exp(-rain[day] * washoff_coeff))
                pesticide_mass_soil[day] += canopy_mass - pesticide_remaining
                last_application = day  # JCH - sure?
    return pesticide_mass_soil


@njit
def find_node(n, depth, target_depth):
    n -= 1  # Zero indexing
    if target_depth >= depth[n]:
        node = n - 1
    elif target_depth <= depth[1]:
        node = 0
    else:
        for node in range(depth.size):
            if target_depth <= depth[node]:
                break
        # select the node that's closer to the depth
        if depth[node] - target_depth > target_depth - depth[node - 1]:
            node -= 1
    return node


@njit
def initialize_soil(plant_factor, cn_crop, cn_fallow, usle_c_crop, usle_c_fallow, bd_5, fc_5, wp_5, bd_20, fc_20, wp_20,
                    usle_k, usle_ls, usle_p, increments_1, increments_2, delta_x, cn_min):
    # Interpolate curve number and c factor based on canopy coverage
    cn_daily = cn_fallow + (plant_factor * (cn_crop - cn_fallow))
    usle_c_daily = usle_c_fallow + (plant_factor * (usle_c_crop - usle_c_fallow))
    for i in range(cn_daily.size):
        if cn_daily[i] <= 0:
            cn_daily[i] = cn_min  # TODO - this shouldn't be necessary. Set cn bounds elsewhere

    # USLE K, LS, C, P factors are multiplied together to estimate erosion losses.
    # Source: PRZM5 Manual Section 4.10 (Young and Fry, 2016)
    usle_klscp_daily = usle_k * usle_ls * usle_c_daily * usle_p
    # kwfact, uslels are zero

    # Generalize soil properties with depth
    # multiply fc_5, wp_5, fc_20, wp_20 by the thickness (delta_x) to get total water retention capacity for layer
    # not sure why bulk density is here - doesn't provide a meaningful output - NT 8/28/18
    bulk_density = np.zeros(increments_1 + increments_2, dtype=np.float32)
    field_capacity = np.zeros(increments_1 + increments_2, dtype=np.float32)
    wilting_point = np.zeros(increments_1 + increments_2, dtype=np.float32)
    cumulative_depth = np.zeros(increments_1 + increments_2, dtype=np.float32)
    cumulative_depth[0] = delta_x[0]
    for i in range(increments_1):
        bulk_density[i] = bd_5 * delta_x[i]
        field_capacity[i] = fc_5 * delta_x[i]
        wilting_point[i] = wp_5 * delta_x[i]
        if i > 0:
            cumulative_depth[i] = cumulative_depth[i - 1] + delta_x[i]
    for i in range(increments_1, increments_1 + increments_2):
        bulk_density[i] = bd_20 * delta_x[i]
        field_capacity[i] = fc_20 * delta_x[i]
        wilting_point[i] = wp_20 * delta_x[i]
    return cn_daily, bulk_density, field_capacity, wilting_point, cumulative_depth, usle_klscp_daily


def partition_benthic(erosion, erosion_mass, surface_area):
    """ Compute concentration in the benthic layer based on mass of eroded sediment """

    from parameters import benthic

    soil_volume = benthic.depth * surface_area
    pore_water_volume = soil_volume * benthic.porosity
    benthic_mass = benthic_loop(erosion, erosion_mass, soil_volume)
    return benthic_mass / pore_water_volume


def pesticide_to_field(applications, new_years, event_dates, rain):
    application_mass = np.zeros((2, rain.size))  # canopy/ground
    for i in range(applications.shape[0]):  # n applications
        crop, event, offset, canopy, step, window1, pct1, window2, pct2, effic, rate = applications[i]
        event_date = int(event_dates[int(event)])
        daily_mass_1 = rate * effic * (pct1 / 100.) / window1
        for year in range(new_years.size):  # n years
            new_year = new_years[year]
            for k in range(int(window1)):
                date = int(new_year + event_date + offset + k)
                application_mass[int(canopy), date] = daily_mass_1
            if step:
                daily_mass_2 = rate * effic * (pct2 / 100.) / window2
                for l in range(window2):
                    date = int(new_year + event_date + window1 + offset + l)
                    application_mass[int(canopy), date] = daily_mass_2
    return application_mass


def plant_growth(n_dates, new_year, plant_begin, harvest_begin, bogey=None):
    # TODO - there may be a faster way to do this. Refer to my question on StackOverflow
    plant_factor = np.zeros(n_dates + 366)
    emergence_begin = int(plant_begin) + 7
    maturity_begin = int((emergence_begin + harvest_begin) / 2)

    if maturity_begin > emergence_begin > 0:
        growth_period = (maturity_begin - emergence_begin) + 1
        mature_period = (harvest_begin - maturity_begin) + 1
        emergence_dates = (new_year + emergence_begin).astype(np.int16)
        maturity_dates = (new_year + maturity_begin).astype(np.int16)
        try:
            growth_dates = np.add.outer(emergence_dates, np.arange(growth_period))
            mature_dates = np.add.outer(maturity_dates, np.arange(mature_period))
            plant_factor[growth_dates] = np.linspace(0, 1, growth_period)
            plant_factor[mature_dates] = 1
        except Exception as e:
            input("Get ready!")
            print(444, plant_begin, emergence_begin, maturity_begin, harvest_begin)
            raise e
        # plant_factor[] = 1
    else:
        emergence_begin = maturity_begin = 0
    return plant_factor[:n_dates], emergence_begin, maturity_begin


@njit
def process_erosion(slope, manning_n, runoff, rain, cn, usle_klscp, raintype, flag):
    # Initialize output
    erosion_loss = np.zeros(rain.size)
    l_sheet = min(100. * np.sqrt(slope) / manning_n, 100.)
    l_shallow = (100. * np.sqrt(slope) / manning_n) - l_sheet

    for day in range(runoff.size):
        if runoff[day] > 0.:
            if slope > 0:
                t_conc_sheet = (0.007 * (manning_n * l_sheet) ** 0.8) / (np.sqrt(rain[day] / 0.0254) * (slope ** 0.4))
                t_conc_shallow = l_shallow / 58084.2 / np.sqrt(slope)
                t_conc = t_conc_sheet + t_conc_shallow
            else:
                t_conc = 0

            # Calculate potential maximum retention after runoff begins (ia_over_p = S in NRCS curve number equa.)
            # PRZM5 Manual (Young and Fry, 2016), Sect 4.6; TR-55 (USDA NRCS, 1986), Chapter 2 """
            if rain[day] > 0:
                ia_over_p = .0254 * (200. / cn[day] - 2.) / rain[day]  # 0.2 * s, in inches
            else:
                ia_over_p = 0

            # lower and upper limit of applicability of NRCS Curve Number method according to TR-55
            # Source: PRZM5 Manual (Young and Fry, 2016), Sect 4.6; TR-55 (USDA NRCS, 1986), Chapter 2
            c = np.zeros(raintype.shape[1])
            if ia_over_p <= 0.1:
                c[:] = raintype[0]
            elif ia_over_p >= 0.5:
                c[:] = raintype[-1]
            else:  # interpolation of intermediate. clunky because numba
                lower = (20. * (ia_over_p - 0.05)) - 1
                delta = raintype[int(lower) + 1] - raintype[int(lower)]
                interp = (lower % 1) * delta
                c[:] = raintype[int(lower)] + interp

            # Calculation of unit peak discharge, peak storm runoff (qp) and erosion loss (on a unit area basis)
            # Coefficients c[0], c[1], c[2] are based on rainfall type, from tr_55.csv
            # Source: PRZM5 Manual (Young and Fry, 2016), Sect 4.10
            peak_discharge = 10. ** (c[0] + c[1] * np.log10(t_conc) + c[2] * (np.log10(t_conc)) ** 2)
            qp = 1.54958679 * runoff[day] * peak_discharge
            erosion_loss[day] = 1.586 * (runoff[day] * 1000. * qp) ** .56 * usle_klscp[day] * 1000.  # kg/d
            if flag:
                print(day, runoff[day], peak_discharge, t_conc, np.log10(t_conc), usle_klscp[day])

    return erosion_loss


@njit
def rain_and_snow(precip, temp, sfac):
    rain_and_melt = np.zeros((2, precip.size))
    snow_accumulation = 0.
    for i in range(precip.size):
        if temp[i] <= 0:
            snow_accumulation += precip[i]
        else:
            rain_and_melt[0, i] = precip[i]
            snow_melt = min(snow_accumulation, sfac / 100 * temp[i])  # convert sfac from cm/deg C/da to m/deg C/da
            snow_accumulation -= snow_melt
            rain_and_melt[1, i] = precip[i] + snow_melt
    return rain_and_melt


def soil_to_water(pesticide_mass_soil, runoff, erosion, leaching, bulk_density, soil_water, kd, deg_soil,
                  runoff_effic, erosion_effic, delta_x, soil_depth):
    # Initialize running variables
    transported_mass = np.zeros((2, pesticide_mass_soil.size), dtype=np.float32)  # runoff, erosion
    total_mass, degradation_rate = 0, 0

    # Initialize erosion intensity
    erosion_intensity = erosion_effic / soil_depth

    # use degradation rate based on degradation in soil (deg_soil) - NT 8/28/18
    for day in range(pesticide_mass_soil.size):
        daily_runoff = runoff[day] * runoff_effic
        total_mass = total_mass * degradation_rate + pesticide_mass_soil[day]
        retardation = (soil_water[day] / delta_x) + (bulk_density * kd)
        deg_total = deg_soil + ((daily_runoff + leaching[day]) / (delta_x * retardation))
        if leaching[day] > 0:
            degradation_rate = np.exp(-deg_total)
        else:
            degradation_rate = np.exp(-deg_soil)

        average_conc = ((total_mass / retardation / delta_x) / deg_total) * (1 - degradation_rate)

        if runoff[day] > 0:
            transported_mass[0, day] = average_conc * daily_runoff  # runoff
        if erosion[day] > 0:
            enrich = np.exp(2.0 - (0.2 * np.log10(erosion[day])))
            enriched_eroded_mass = erosion[day] * enrich * kd * erosion_intensity * 0.1
            transported_mass[1, day] = average_conc * enriched_eroded_mass
    return transported_mass


@dask.delayed
def stage_one_to_two(precip, pet, temp, new_year,
                     covmax, orgC_5, bd_5, overlay, plant_begin, bloom_begin, harvest_begin, cn_cov,
                     cn_fallow, usle_c_cov, usle_c_fal, fc_5, wp_5, bd_20, fc_20, wp_20, kwfact, usle_ls, usle_p,
                     irr_type, anetd, amxdr, cintcp, rainfall, slope, mannings_n, flag,
                     scenario_id=None):
    from parameters import soil, types, sfac

    type_matrix = types[types.index == int(rainfall)].values.astype(np.float32)

    # Model the growth of plant between emergence and maturity (defined as full canopy cover)
    plant_factor, emergence_begin, maturity_begin = \
        plant_growth(precip.size, new_year, plant_begin, harvest_begin, scenario_id)

    # Initialize soil properties for depth
    cn, bulk_density, field_capacity, wilting_point, cumulative_depth, usle_klscp = \
        initialize_soil(plant_factor, cn_cov, cn_fallow, usle_c_cov, usle_c_fal, bd_5, fc_5, wp_5, bd_20, fc_20,
                        wp_20, kwfact, usle_ls, usle_p, soil.increments_1, soil.increments_2, soil.delta_x, soil.cn_min)

    # Simulate surface hydrology
    rain_and_effective = rain_and_snow(precip, temp, sfac)
    rain, effective_rain = rain_and_effective[0], rain_and_effective[1]  # clunky unpacking for numba
    runoff, soil_water, leaching = \
        surface_hydrology(field_capacity, wilting_point, plant_factor, cn, cumulative_depth,
                          irr_type, soil.deplallw, anetd, amxdr, soil.leachfrac, cintcp, effective_rain, rain, pet,
                          soil.increments_1, soil.increments_2, soil.delta_x)

    # Calculate erosion loss
    erosion = process_erosion(slope, mannings_n, runoff, effective_rain, cn, usle_klscp, type_matrix, flag)

    # If scenario is an overlay, set runoff and erosion to zero. # This applies to CDL values identified as
    # double-cropped. See double_crops in parameters.py
    arrays = np.float32([runoff, erosion, leaching, soil_water, rain])
    vars = np.float32(
        [covmax, orgC_5, bd_5, overlay, plant_begin, emergence_begin, bloom_begin, maturity_begin, harvest_begin])

    return arrays, vars


@njit
def surface_hydrology(field_capacity, wilting_point, plant_factor, cn, depth,  # From other function output
                      irrigation_type, irr_depletion, anetd, root_max, leaching_factor, cintcp,  # From scenario
                      effective_rain, rain, potential_et,  # From metfile
                      increments_1, increments_2, delta_x):  # From parameters

    """ Process hydrology parameters, returning daily runoff, soil water content, runoff velocity """
    # Initialize arrays and running variables
    # By day
    n_dates = plant_factor.size
    daily_velocity = np.zeros(n_dates, dtype=np.float32)
    daily_soil_water = np.zeros(n_dates, dtype=np.float32)
    runoff = np.zeros(n_dates, dtype=np.float32)
    # By node
    n_soil_increments = increments_1 + increments_2
    soil_layer_loss = np.zeros(n_soil_increments, dtype=np.float32)
    velocity = np.zeros(n_soil_increments, dtype=np.float32)
    et_factor = np.zeros(n_soil_increments, dtype=np.float32)
    available_water = np.zeros(n_soil_increments, dtype=np.float32)
    soil_water = field_capacity.copy()

    # Running variable
    canopy_holdup = 0

    # Calculate total available water (field) capacity, irrigation trigger based on rooting depth
    # Source: PRZM5 Manual, Section 4.4 (Young and Fry, 2016)
    fc_minus_wp = field_capacity - wilting_point
    if irrigation_type > 0:
        irrigation_node = find_node(n_soil_increments, depth, root_max)
        target_dryness = 0
        for i in range(irrigation_node):
            target_dryness += fc_minus_wp[i] * irr_depletion + wilting_point[i]
        total_fc = np.sum(field_capacity[:irrigation_node])

    # Set evaporation node
    evaporation_node = find_node(n_soil_increments, depth, anetd)  # node only for evaporation

    for day in range(rain.size):

        """
        s variable for soil moisture curve number; potential maximum retention. units are in meters
        Source:  PRZM5 Manual Section 4.6 (Young and Fry, 2016); USDA NRCS (1986) TR-55, Chapter 2
        """
        s = (25.4 / cn[day]) - .254

        """
        Process irrigation and modify effective_rain, as needed
        Source:  PRZM5 Manual Section 4.4 (Young and Fry, 2016)
        """
        overcanopy_irrigation = 0
        if irrigation_type > 0:
            current_dryness = np.sum(soil_water[:irrigation_node])
            daily_max_irrigation = 0.2 * s
            if current_dryness < target_dryness and effective_rain[day] <= 0.:
                irrig_required = (total_fc - current_dryness) * leaching_factor + 1.
            if irrigation_type == 3:
                overcanopy_irrigation = min(irrig_required, daily_max_irrigation)
                effective_rain[day] = overcanopy_irrigation
            elif irrigation_type == 4:  # undercanopy irrigation
                effective_rain[day] = min(irrig_required, daily_max_irrigation)

        """ Determine daily runoff (Soil moisture curve number option) """
        if effective_rain[day] > (0.2 * s):  # runoff by the Curve Number Method
            # print(day, effective_rain[day], cn[day], s)
            runoff[day] = max(0, (effective_rain[day] - (0.2 * s)) ** 2 / (effective_rain[day] + (0.8 * s)))

        """ 
        Leaching and canopy holdup
        Upper bound for leaching is based on effective rain (precipitation, irrigation, melted snow,
        net canopy interception/gain) minus runoff
        Source: PRZM5 Manual Section 4.9 (Young and Fry, 2016)
        """
        leaching = effective_rain[day] - runoff[day]
        if rain[day] > 0. or overcanopy_irrigation > 0:
            available_canopy_gain = (rain[day] + overcanopy_irrigation) * (1. - runoff[day] / effective_rain[day])
            delta_water = min(available_canopy_gain, cintcp * plant_factor[day] - canopy_holdup)
            canopy_holdup += delta_water
            leaching -= delta_water
        et_from_canopy = canopy_holdup - potential_et[day]
        canopy_holdup = max(0., et_from_canopy)

        """ 
        Calculate soil layer loss from ET
        If soil moisture is < 0.6 of available water, available soil ET is reduced proportionally to wp
        Source: PRZM5 Manual Section 4.8 (Young and Fry, 2016)
        """
        available_soil_et = max(0., -et_from_canopy)

        # Set ET node and adjust et_depth by maximum root depth, scaled by plant growth factor
        et_node = evaporation_node
        if plant_factor[day] > 0:
            et_depth = plant_factor[i] * root_max
            if et_depth > anetd:
                et_node = find_node(n_soil_increments, depth, et_depth)

        check_moisture_et, target_moisture_et = 0., 0.
        for i in range(et_node + 1):
            available_water[i] = soil_water[i] - wilting_point[i]
            check_moisture_et += soil_water[i] - wilting_point[i]
            target_moisture_et += 0.6 * fc_minus_wp[i]
            et_factor[i] = (depth[et_node] - depth[i] + delta_x[i]) * available_water[i]
        if check_moisture_et < target_moisture_et:
            available_soil_et *= check_moisture_et / target_moisture_et
        et_sum = np.sum(et_factor[:et_node + 1])
        if et_sum > 0:
            for i in range(et_node + 1):
                et_factor[i] /= et_sum
        else:
            et_factor[:] = 0.
        for i in range(et_node + 1):
            soil_layer_loss[i] = available_soil_et * et_factor[i]  # potential loss

        """
        Leaching loop. "velocity" is derived from excess water after runoff
        Source: PRZM5 Manual Section 4.9 (Young and Fry, 2016)
        """

        last_velocity = leaching
        for node in range(n_soil_increments):
            water_level = last_velocity - soil_layer_loss[node] + soil_water[node]

            # If the amount of water coming down exceeds field capacity, some of it passes through
            if water_level > field_capacity[node]:
                velocity[node] = water_level - field_capacity[node]
                soil_water[node] = field_capacity[node]

            # If the amount coming down is less than field capacity, it stays in the soil
            else:
                velocity[node] = 0.
                soil_water[node] = max(water_level, wilting_point[node])

            if velocity[node] <= 0. and node > et_node:
                velocity[node:n_soil_increments] = 0.
                break

            last_velocity = velocity[node]

        daily_velocity[day] = velocity[0]
        daily_soil_water[day] = soil_water[0]

    return runoff, daily_soil_water, daily_velocity
