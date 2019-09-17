import os
import re
import dask
import numpy as np
import pandas as pd

from utilities import DateManager, HydroTable, Navigator, ImpulseResponseMatrix, MemoryMatrix, report
from paths import weather_path, stage_one_scenario_path, stage_two_scenario_path, recipe_path
from parameters import hydrology


class Reaches(DateManager):
    def __init__(self, scenarios, recipes, region, output, progress_interval=10000):
        self.output = output
        self.scenarios = scenarios
        self.recipes = recipes
        self.region = region
        self.progress_interval = progress_interval

        # Initialize dates
        DateManager.__init__(self, scenarios.start_date, scenarios.end_date)

        self.recipe_ids = sorted(region.all_reaches)
        self.active_reaches = region.active_reaches
        self.outlets = set(self.recipe_ids) & set(self.region.lake_table.outlet_comid)

        # Keep track of which reaches have been run
        self.burned_reaches = set()  # reaches that have been analyzed

        # Initialize local matrix: matrix of local runoff and mass, for rapid internal recall
        self.local = MemoryMatrix([self.recipe_ids, 2, self.n_dates], name='local scenario')

        # Discharge from lakes
        self.lake_discharge = MemoryMatrix([list(self.outlets), 2, self.n_dates], name='lake discharge')

        # Initialize a matrix to store benthic concentrations, since these are only performed locally
        self.benthic = MemoryMatrix([self.recipe_ids, self.n_dates], name='benthic concentrations')

    def burn_lake(self, lake):
        from parameters import hydrology as hydrology_params

        irf = ImpulseResponseMatrix.generate(1, lake.residence_time, self.n_dates)

        # Get the convolution function
        # Get mass and runoff for the reach
        total_mass, total_runoff = self.upstream_loading(lake.outlet_comid)

        # Modify combined time series to reflect lake
        new_mass = np.convolve(total_mass, irf)[:self.n_dates]
        if hydrology_params.convolve_runoff:  # Convolve runoff
            new_runoff = np.convolve(total_runoff, irf)[:self.n_dates]
        else:  # Flatten runoff
            new_runoff = np.repeat(np.mean(total_runoff), self.n_dates)

        # Add all lake mass and runoff to outlet
        self.lake_discharge.update(lake.outlet_comid, np.array([new_mass, new_runoff]))

    def combine_scenarios(self, reach_id):
        """  Fetch all scenarios and multiply by area. For erosion, area is adjusted. """
        scenario_ids, areas = self.recipes.fetch(reach_id)

        if scenario_ids is not None:
            # Fetch time series data from processed scenarios
            scenario_data, indices = self.scenarios.fetch(scenario_ids)  # runoff, runoff_mass, erosion, erosion_mass

            # Reshape matrix to (n_vars, n_dates, n_scenarios)
            scenario_data = scenario_data.swapaxes(1, 2)

            # Modify values based on area of scenario
            areas = areas[indices].astype(np.float32)  # (n_scenarios,)
            erosion_modifier = np.power(areas / 10000., .12)  # (n_scenarios)
            scenario_data[:2] *= areas  # runoff, runoff_mass
            scenario_data[2:] *= erosion_modifier  # erosion, erosion_mass

            # Add up all scenarios
            return scenario_data.sum(axis=2)  # [scenarios, vars, n_dates]
        else:
            return np.zeros((4, self.n_dates), dtype=np.float32)

    def process_local(self, reach_id, verbose=False):
        from model_functions import partition_benthic

        # Fetch time series data from all scenarios in recipe
        time_series_data = self.combine_scenarios(reach_id)

        if time_series_data is not None:

            runoff, runoff_mass, erosion, erosion_mass = time_series_data

            # Assess the contributions to the recipe from ach source (runoff/erosion) and crop
            # self.o.update_contributions(recipe_id, scenarios, time_series[[1, 3]].sum(axis=1))

            # Run benthic/water column partitioning if active
            if reach_id in self.active_reaches:
                surface_area = self.region.flow_file.fetch(reach_id)["surface_area"]
                benthic_conc = partition_benthic(erosion, erosion_mass, surface_area)
            else:
                benthic_conc = np.zeros(self.n_dates)

            # Update local array with mass and runoff
            self.local.update(reach_id, np.array([runoff_mass, runoff]))
            self.benthic.update(reach_id, benthic_conc)

        elif verbose:
            report("No scenarios found for {}".format(reach_id))

    def process_upstream(self, reach_id):

        from model_functions import compute_concentration

        # Process upstream contributions
        mass, runoff = self.upstream_loading(reach_id)

        flow = self.region.flow_file.flows(reach_id, self.month_index)

        total_flow, (concentration, runoff_conc) = \
            compute_concentration(mass, runoff, self.n_dates, flow)

        # Calculate exceedances
        self.output.update_exceedances(reach_id, concentration)

        benthic_conc = self.benthic.fetch(reach_id)

        # Store results in output array
        self.output.update_time_series(reach_id, total_flow, runoff, mass, concentration, benthic_conc)

    def upstream_loading(self, reach_id):
        """ Identify all upstream reaches, pull data and offset in time """
        from parameters import hydrology as hydrology_params

        # Fetch all upstream reaches and corresponding travel times
        upstream_reaches, travel_times, warning = \
            self.region.nav.upstream_watershed(reach_id, return_times=True, return_warning=True)

        # Filter out reaches (and corresponding times) that have already been burned
        indices = np.int16([i for i, r in enumerate(upstream_reaches) if r not in self.burned_reaches])
        reaches, reach_times = upstream_reaches[indices], travel_times[indices]

        # Determine which of the reaches correspond to lake outlets
        outlet_indices = np.int16([i for i, r in enumerate(reaches) if r in self.outlets])
        outlets, outlet_times = reaches[outlet_indices], reach_times[outlet_indices]

        # Don't need to do proceed if there's nothing upstream
        if len(reaches) > 1 or len(outlets) > 0:

            # Initialize the output array
            total_mass_and_runoff = np.zeros((2, self.n_dates))  # (mass/runoff, dates)

            # Fetch time series data for each upstream reach
            reach_array = self.local.fetch(reaches)  # (reaches, vars, dates)

            # Add in outlets
            if outlets.any():
                outlet_array = self.lake_discharge.fetch(outlets)  # (outlets, vars, dates)
                reach_array = np.concatenate([reach_array, outlet_array])
                reach_times = np.concatenate([reach_times, outlet_times])

            for tank in range(np.max(reach_times) + 1):
                in_tank = reach_array[reach_times == tank].sum(axis=0)
                if tank > 0:
                    if hydrology_params.gamma_convolve:
                        irf = self.region.irf.fetch(tank)  # Get the convolution function
                        in_tank[0] = np.convolve(in_tank[0], irf)[:self.n_dates]  # mass
                        in_tank[1] = np.convolve(in_tank[1], irf)[:self.n_dates]  # runoff
                    else:
                        in_tank = np.pad(in_tank[:, :-tank], ((0, 0), (tank, 0)), mode='constant')
                total_mass_and_runoff += in_tank  # Add the convolved tank time series to the total for the reach

            mass, runoff = total_mass_and_runoff
        else:
            mass, runoff = self.local.fetch(reach_id)

        return mass, runoff


class InputParams(DateManager):
    """
    User-specified parameters and parameters derived from hem.
    This class is used to hold parameters and small datasets that are global in nature and apply to the entire model
    run including Endpoints, Crops, Dates, Intake reaches, and Impulse Response Functions
    """

    def __init__(self, input_dict):

        # Read input dictionary
        self.__dict__.update(input_dict)

        # Read endpoints and applications
        self.endpoints = self.read_endpoints()
        self.applications = self.read_applications()
        self.crops = set(self.applications.crop)

        # Dates
        DateManager.__init__(self, np.datetime64(self.sim_date_start), np.datetime64(self.sim_date_end))

        # Processing extent
        self.intakes, self.active_regions, self.active_reaches = self.processing_extent()

        # Initialize an impulse response matrix if convolving timesheds
        self.irf = None if not hydrology.gamma_convolve else ImpulseResponseMatrix(self.dates.size)

        # Make numerical adjustments (units etc)
        self.adjust_data()

        # Read token
        self.token = \
            self.simulation_name if not hasattr(self, 'csrfmiddlewaretoken') else self.csrfmiddlewaretoken

    def adjust_data(self):
        """ Convert half-lives to degradation rates """
        # NOTE that not all half-life inputs are included below: 'aqueous' should not be used for 'soil'
        # soil_hl applies to the half-life of the pesticide in the soil/on the field
        # wc_metabolism_hl applies to the pesticide when it reaches the water column
        # ben_metabolism_hl applies to the pesticide that is sorbed to sediment in the benthic layer of water body
        # aq_photolysis_hl and hydrolysis_hl are abiotic degradation routes that occur when metabolism is stable
        # Following the input naming convention in fields_and_qc.csv for the 'old', here is a suggested revision:
        # for old, new in [('soil_hl', 'deg_soil'), ('wc_metabolism_hl', 'deg_wc_metabolism'),
        #                  ('ben_metabolism_hl', 'deg_ben_metabolism'), ('photolysis_hl', 'aq_photolysis'),
        #                 ('hydrolysis_hl', 'hydrolysis')] - NT: 8/28/18

        adjust = lambda x: 0.693 / x if x else np.inf  # half-life of 'zero' indicates stability
        for old, new in [('aqueous', 'soil'), ('photolysis', 'aq_photolysis'),
                         ('hydrolysis', 'hydrolysis'), ('wc', 'wc_metabolism')]:
            setattr(self, 'deg_{}'.format(old), adjust(getattr(self, "{}_hl".format(new))))

        self.applications.rate *= 0.0001  # convert kg/ha -> kg/m2 (1 ha = 10,000 m2)

    def read_applications(self):
        from utilities import fields
        from parameters import input_indices

        application_fields = fields.fetch('applications')
        header_order = input_indices['application']

        if len(set(application_fields) & set(header_order)) != len(application_fields):
            raise Exception('"Mismatch between application fields in table and script"')
        return pd.DataFrame(self.applications, columns=application_fields)[header_order]

    def read_endpoints(self):
        from utilities import endpoint_format

        endpoints = pd.DataFrame(self.endpoints.T, columns=('acute_tox', 'chronic_tox', 'overall_tox'))
        return pd.concat([endpoint_format, endpoints], axis=1)

    def processing_extent(self):
        """ Determine which NHD regions need to be run to process the specified reacches """
        from parameters import nhd_regions
        from paths import dwi_path, manual_points_path

        assert self.sim_type in ('eco', 'drinking_water', 'manual'), \
            "Invalid simulation type '{}'".format(self.sim_type)

        # Get the path of the table used for intakes
        intake_file = {'drinking_water': dwi_path, 'manual': manual_points_path}.get(self.sim_type)

        # Get reaches and regions to be processed if not running eco
        if intake_file is not None:
            intakes = pd.read_csv(intake_file)
            intakes['Region'] = [str(region).zfill(2) for region in intakes.Region]
            try:
                active_regions = sorted(np.unique(intakes.Region))
                active_reaches = sorted(np.unique(intakes.COMID))
            except AttributeError as e:
                raise Exception("Invalid intake table: must have 'Region' and 'COMID' columns") from e

        else:  # Run everything if running eco
            intakes = active_reaches = None
            active_regions = sorted(nhd_regions)

        return intakes, active_regions, active_reaches


class Outputs(DateManager):
    def __init__(self, i, scenario_ids, start_date, end_date):
        from paths import output_path

        self.input = i
        self.recipe_ids = sorted(i.active_reaches)
        self.scenario_ids = scenario_ids
        self.output_dir = os.path.join(output_path, i.token)
        self.write_list = self.input.active_reaches

        DateManager.__init__(self, start_date, end_date)

        # Initialize output JSON dict
        self.out_json = {}

        # Initialize output matrices
        self.output_fields = ['total_flow', 'total_runoff', 'total_mass', 'total_conc', 'benthic_conc']
        self.time_series = MemoryMatrix([sorted(self.write_list), self.output_fields, self.n_dates],
                                        name='output time series')

        # Initialize exceedances matrix: the probability that concentration exceeds endpoint thresholds
        self.exceedances = MemoryMatrix([self.recipe_ids, self.input.endpoints.shape[0], 3], name='exceedance')

        # Initialize contributions matrix: loading data broken down by crop and runoff v. erosion source
        self.contributions = MemoryMatrix([2, self.recipe_ids, self.input.crops], name='contributions')
        self.contributions.columns = np.int32(sorted(self.input.crops))
        self.contributions.header = ["cls" + str(c) for c in self.contributions.columns]

    def update_contributions(self, recipe_id, scenario_names, loads):
        """ Sum the total contribution by land cover class and add to running total """
        classes = [int(re.match('s[A-Za-z\d]{10,12}w\d{2,8}lc(\d+?)', name).group(1)) for name in scenario_names]
        contributions = np.zeros((2, 255))
        for i in range(2):  # Runoff Mass, Erosion Mass
            contributions[i] += np.bincount(classes, weights=loads[i], minlength=255)

        self.contributions.update(recipe_id, contributions[:, self.contributions.columns])

    def update_exceedances(self, recipe_id, concentration):
        from model_functions import exceedance_probability

        # Extract exceedance durations and corresponding thresholds from endpoints table
        durations = \
            np.int16(self.input.endpoints[['acute_duration', 'chronic_duration', 'overall_duration']].as_matrix())
        thresholds = \
            np.int16(self.input.endpoints[['acute_tox', 'chronic_tox', 'overall_tox']].as_matrix())

        # Calculate excedance probabilities
        exceed = exceedance_probability(concentration, durations.flatten(), thresholds.flatten(), self.year_index)

        self.exceedances.update(recipe_id, exceed.reshape(durations.shape))

    def update_time_series(self, recipe_id, total_flow=None, total_runoff=None, total_mass=None, total_conc=None,
                           benthic_conc=None):

        self.time_series.update(recipe_id, np.vstack([total_flow, total_runoff, total_mass, total_conc, benthic_conc]))

    def write_json(self, write_exceedances=False, write_contributions=False):

        encoder.FLOAT_REPR = lambda o: format(o, '.4f')
        out_file = os.path.join(self.output_dir, "{}_json.csv".format(self.input.chemical_name))
        out_json = {"COMID": {}}
        for recipe_id in self.recipe_ids:
            out_json["COMID"][str(recipe_id)] = {}
            if write_exceedances:
                labels = ["{}_{}".format(species, level)
                          for species in self.input.endpoints.species for level in ('acute', 'chronic', 'overall')]
                exceedance_dict = dict(zip(labels, np.float64(self.exceedances.fetch(recipe_id)).flatten()))
                out_json["COMID"][str(recipe_id)].update(exceedance_dict)
            if write_contributions:
                contributions = self.contributions.fetch(recipe_id)
                for i, category in enumerate(("runoff", "erosion")):
                    labels = ["{}_load_{}".format(category, label) for label in self.contributions.header]
                    contribution_dict = dict(zip(labels, np.float64(contributions[i])))
                    out_json["COMID"][str(recipe_id)].update(contribution_dict)

        out_json = json.dumps(dict(out_json), sort_keys=True, indent=4, separators=(',', ': '))
        with open(out_file, 'w') as f:
            f.write(out_json)

    def write_output(self):
        from parameters import write_contributions, write_exceedances

        # Create output directory
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # Write JSON output
        self.write_json(write_exceedances, write_contributions)

        # Write time series
        if self.write_time_series:
            self.write_time_series()

    def write_time_series(self, fields='all'):
        from parameters import heading_lookup

        if fields == 'all':
            fields = self.output_fields
            field_indices = np.arange(len(self.output_fields))
        else:
            field_indices = [self.output_fields.index(field) for field in fields]

        headings = [heading_lookup.get(field, "N/A") for field in fields]

        for recipe_id in self.write_list:
            out_file = os.path.join(self.output_dir, "time_series_{}.csv".format(recipe_id))
            out_data = self.time_series.fetch(recipe_id)[field_indices].T
            df = pd.DataFrame(data=out_data, index=self.dates, columns=headings)
            df.to_csv(out_file)


class Recipes(object):
    def __init__(self, region, year):
        self.region = region
        self.year = year
        self.path = recipe_path.format(region, year)

        # Load recipes
        data = np.load(self.path)
        self.matrix = data['recipes']
        self.map = data['map']

    def fetch(self, reach_id):
        try:
            start, end = self.map[self.map[:, 0] == reach_id].flat[1:].astype(np.int32)
            return self.matrix[start:end].T
        except ValueError:
            report("No recipes found for reach {}".format(reach_id))
            return None, None


class StageOneScenarios(object):
    def __init__(self, region):
        self.path = stage_one_scenario_path.format(region)
        self._scenario_id = None
        self._default_dates = None

    @property
    def temporary_dates_fix(self):
        """
        Replace missing or invalid dates with averages. This should be fixed upstream from here
        """
        if self._default_dates is None:
            date_fields = ['plant_begin', 'plant_begin_active', 'emergence_begin', 'maxcover_begin',
                           'harvest_begin', 'harvest_begin_active', 'plant_end', 'plant_end_active',
                           'emergence_end', 'maxcover_end', 'harvest_end', 'harvest_end_active']
            dates = pd.read_csv(self.path, usecols=date_fields)
            for field in date_fields:
                dates[(dates[field] < 0) | (dates[field] > 366)] = np.nan
            self._default_dates = dict(zip(date_fields, dates.mean(axis=0).astype(np.int32)))
        return self._default_dates

    @property
    def scenario_ids(self):
        if self._scenario_id is None:
            self._scenario_id = pd.read_csv(self.path, usecols=['scenario_id']).values.T[0]
        return self._scenario_id

    def modify_array(self, array):
        # TODO - put all of this in aquatic-model-inputs workflow?
        for var in ('orgC_5', 'cintcp', 'slope', 'covmax', 'anetd', 'amxdr'):
            array[var] /= 100.  # cm -> m
        for var in ('anetd', 'amxdr'):
            array[var] = np.min((array[var], array['root_zone_max'] / 100.))
        for var in ['bd_5', 'bd_20']:
            array[var] *= 1000  # kg/m3
        array.loc[array.kwfact == 0, 'kwfact'] = 0.2  # TODO - Why are so many zeros?
        array.loc[array.usle_p == 0, 'usle_p'] = 0.25  # TODO - same here
        array.loc[array.usle_ls == 0, 'usle_ls'] = 1.0  # TODO - same here. Scenarios issue I guess.
        array['rainfall'] = (array.rainfall
                             .map({0.0: 0, 'I': 0, 'IA': 1, 'II': 2, 'III': 3})
                             .fillna(0)
                             .astype(np.float32))
        array['irrigation_type'] = array.irrigation_type.map({'over': 1, 'under': 2})
        for field, default_val in self.temporary_dates_fix.items():
            sel = pd.isnull(array[field]) | (array[field] < 0) | (array[field] > 732)
            array.loc[sel, field] = default_val
            array[field] = array[field].astype(np.int32)
        return array

    def iterate(self):
        from parameters import stage_one_chunksize

        for chunk in pd.read_csv(self.path, chunksize=stage_one_chunksize):
            chunk = self.modify_array(chunk)
            for weather_grid, scenarios in chunk.groupby('weather_grid'):
                yield weather_grid, scenarios

    @property
    def n_scenarios(self):
        return len(self.scenario_ids)


class StageTwoScenarios(DateManager):
    def __init__(self, region, met=None, sim_start_date=None, sim_end_date=None):
        # self.scenario_ids = stage_one.scenario_ids
        self.path = stage_two_scenario_path.format(region)
        self.stage_one = StageOneScenarios(region)
        self.met = met
        self.keyfile_path = self.path + "_key.txt"
        self.array_path = self.path + "_arrays.dat"
        self.var_path = self.path + "_vars.dat"

        self._scenario_ids = None

        # If the stage 2 scenarios already exist, load the data
        if self.exists:

            self.arrays, self.variables, self.names, self.start_date, time_series_shape, variables_shape = \
                self.load_key()
            self.end_date = self.start_date + time_series_shape[2]
            DateManager.__init__(self, self.start_date, self.end_date)

            # self.start_offset, self.end_offset = self.date_offsets(sim_start_date, sim_end_date)
            self.start_offset = 0
            self.end_offset = 0

            # Load input matrices
            self.time_series_matrix = MemoryMatrix([self.names, self.arrays, self.n_dates],
                                                   path=self.array_path, existing=True, name='scenario')

            self.variable_matrix = MemoryMatrix([self.names, self.variables],
                                                path=self.var_path, existing=True, name='variable')

        else:
            # These are designated ahead of time in order to allocate memory
            # If adjusting the inclusion or order of variables, make matching adjustment in Scenario.__init__
            self.arrays = ['runoff', 'erosion', 'leaching', 'soil_water', 'rain']
            self.variables = \
                ['covmax', 'org_carbon', 'bulk_density', 'overlay',
                 'plant_begin', 'emergence_begin', 'bloom_begin', 'maturity_begin', 'harvest_begin']

            DateManager.__init__(self, met.start_date, met.end_date)

            # Initalize matrices
            self.array_matrix = MemoryMatrix([self.scenario_ids, self.arrays, self.n_dates],
                                             dtype=np.float32, path=self.path + "_arrays")
            self.variable_matrix = MemoryMatrix([self.scenario_ids, self.variables],
                                                dtype=np.float32, path=self.path + "_vars")

            # Create key
            self.create_keyfile()

    def create_keyfile(self):
        with open(self.keyfile_path, 'w') as f:
            f.write(",".join(self.arrays) + "\n")
            f.write(",".join(self.variables) + "\n")
            f.write(",".join(self.scenario_ids) + "\n")
            f.write(pd.to_datetime(self.start_date).strftime('%Y-%m-%d') + "\n")
            f.write(",".join(map(str, self.array_matrix.shape)) + "\n")
            f.write(",".join(map(str, self.variable_matrix.shape)))

    def date_offsets(self, sim_start, sim_end):
        # Get offset between scenario and simulation start dates
        messages = []
        start_offset, end_offset = 0, 0

        if self.start_date > sim_start:  # scenarios start later than selected start date
            messages.append('start date is earlier')
        else:
            start_offset = (sim_start - self.start_date).astype(int)
        if self.end_date < sim_end:
            messages.append('end date is later')
        else:
            end_offset = (self.end_date - sim_end).astype(int)

        if any(messages):
            report("Simulation {} than available scenario data. Full simulation will not be available".format(
                " and ".join(messages)), warn=1)

        # Change scenario dates to reflect
        self.start_date = self.start_date + np.timedelta64(int(start_offset), 'D')
        self.end_date = self.end_date + np.timedelta64(int(end_offset), 'D')

        # To accomodate python indexing.  array[3:-1] would work but array[3:0] will not
        end_offset = start_offset + self.n_dates + 10 if end_offset == 0 else end_offset

        return start_offset, end_offset

    @property
    def exists(self):
        if self.stage_one is None or self.met is None:
            if all(map(os.path.exists, (self.array_path, self.var_path))):
                return True
            else:
                raise FileNotFoundError("Missing array or variable arrays")
        else:
            return False

    def fetch(self, scenario_id):
        end_offset = self.end_offset if self.end_offset > 0 else self.n_dates + 1
        arrays = self.time_series_matrix.fetch(scenario_id)[:, self.start_offset:end_offset]
        vars = self.variable_matrix.fetch(scenario_id)
        return arrays, vars

    def load_key(self):
        with open(self.keyfile_path) as f:
            time_series = next(f).strip().split(",")
            variables = next(f).strip().split(",")
            scenarios = next(f).strip().split(",")
            start_date = np.datetime64(next(f).strip())
            time_series_shape = [int(val) for val in next(f).strip().split(",")]
            variables_shape = [int(val) for val in next(f).strip().split(",")]
        return time_series, variables, np.array(scenarios), start_date, time_series_shape, variables_shape

    def build_from_stage_one(self):
        from parameters import batch_size
        from model_functions import stage_one_to_two

        n_scenarios = int(self.stage_one.n_scenarios)
        batch = []
        count = 0
        # Group by weather grid to reduce the overhead from fetching met data
        # TODO - confirm that results are written in the correct order by dask
        for weather_grid, scenarios in self.stage_one.iterate():
            precip, pet, temp, *_ = self.met.fetch_station(weather_grid)
            for _, s in scenarios.iterrows():
                count += 1
                scenario = \
                    stage_one_to_two(precip, pet, temp, self.met.new_year, s.covmax, s.orgC_5, s.bd_5,
                                     s.overlay, s.plant_begin, s.bloom_begin, s.harvest_begin, s.cn_cov,
                                     s.cn_fallow, s.usle_c_cov, s.usle_c_fal, s.fc_5, s.wp_5, s.bd_20,
                                     s.fc_20, s.wp_20, s.kwfact, s.usle_ls, s.usle_p, s.irrigation_type,
                                     s.anetd, s.amxdr, s.cintcp, s.rainfall, s.slope, s.mannings_n, 0,
                                     'c' + str(count))
                batch.append(scenario)
                if len(batch) == batch_size or (count + 1) == n_scenarios:
                    report("Processing... {}".format(count))
                    run = dask.delayed()(batch)
                    try:
                        arrays, vars = map(np.array,
                                           zip(*run.compute()))  # result: [(arrays0, vars0), (arrays1, vars1)...]
                        start_pos = (int(count / batch_size) - 1) * batch_size
                        a_writer = self.array_matrix.writer
                        v_writer = self.variable_matrix.writer
                        a_writer[start_pos:start_pos + len(batch)] = arrays
                        v_writer[start_pos:start_pos + len(batch)] = vars
                        del a_writer, v_writer
                    except Exception as e:
                        report(e, warn=3)
                    batch = []


class StageThreeScenarios(DateManager, MemoryMatrix):
    def __init__(self, region_id, crops):
        self.stage_two = StageTwoScenarios(region_id)
        self.names = self.select_scenarios(crops)

        # Set dates
        DateManager.__init__(self, self.stage_two.start_date, self.stage_two.end_date)

        # Initialize memory matrix
        MemoryMatrix.__init__(self, [self.names, 2, self.n_dates], name='pesticide mass')

    def select_scenarios(self, crops):
        selected = []
        for scenario_id in self.stage_two.names:
            crop = float(re.search('LC(\d{1,3})$', scenario_id).group(1))
            if crop in crops:
                selected.append(scenario_id)
        return selected

    def build_from_stage_two(self):
        from model_functions import stage_two_to_three
        from parameters import soil, plant, batch_size

        # Initialize readers and writers
        # Reminder: array_matrix.shape, processed_matrix.shape = (scenario, variable, date)
        time_series_reader = self.stage_two.time_series_matrix.reader
        variable_reader = self.stage_two.variable_matrix.reader

        batch = []
        n_scenarios = len(self.names)

        # Iterate scenarios
        for count, scenario_id in enumerate(self.names):

            # Extract stored data
            leaching, runoff, erosion, soil_water, plant_factor, rain = \
                time_series_reader[count, :, self.start_offset:self.end_offset]
            new_vars = variable_reader[count, :4]  # covmax, org_carbon, bulk_density, overlay
            plant_dates = variable_reader[count, 4:]

            # Get crop ID of scenario and find all associated crops in group
            scenario = \
                stage_two_to_three(self.input.applications[self.input.applications.crop == crop].as_matrix(),
                                   self.new_year, self.input.kd_flag, self.input.koc, self.input.deg_aqueous,
                                   leaching, runoff, erosion, soil_water, plant_factor, rain,
                                   soil.cm_2, soil.delta_x_top_layer, soil.erosion_effic, soil.soil_depth,
                                   plant.deg_foliar, plant.washoff_coeff, soil.runoff_effic, plant_dates,
                                   *new_vars)
            batch.append(scenario)
            if len(batch) == batch_size or (count + 1) == n_scenarios:
                report("Processing... {}".format(count))
                try:
                    arrays = dask.delayed()(batch).compute()
                    start_pos = (int(count / batch_size) - 1) * batch_size
                    writer = self.mass_matrix.writer
                    writer[start_pos:start_pos + len(batch)] = arrays
                    del writer
                except Exception as e:
                    report(e, warn=3)
                batch = []

        del time_series_reader, variable_reader


class WeatherArray(MemoryMatrix, DateManager):
    def __init__(self):
        self.path = weather_path
        self.key_path = os.path.join(self.path, "key.csv")
        points_path = os.path.join(self.path, "weather_grid.csv")
        matrix_path = os.path.join(self.path, "weather_cube.dat")

        # Set row/column offsets
        start_date, end_date, self.header = self.load_key()

        # Get station IDs
        self.points = pd.read_csv(points_path).set_index('weather_grid')

        # Set dates
        DateManager.__init__(self, start_date, end_date)

        # Initialize memory matrix
        MemoryMatrix.__init__(self, [self.points.index, self.n_dates, self.header],
                              dtype=np.float32, path=matrix_path, existing=True)

    def load_key(self):
        with open(self.key_path) as f:
            years = sorted(map(int, next(f).split(",")))
            header = next(f).split(",")
        start_date = np.datetime64('{}-01-01'.format(years[0]))
        end_date = np.datetime64('{}-12-31'.format(years[-1]))
        return start_date, end_date, header

    def fetch_station(self, station_id):
        try:
            data = np.array(self.fetch(station_id, copy=True, verbose=True)).T
        except Exception as e:
            report("Met station {} not found".format(station_id), warn=2)
            return
        data[:2] /= 100.  # Precip, PET  cm -> m
        return data


class WatershedHydrology(object):
    """
    Contains all datasets and functions related to the NHD Plus region, including all hydrological features and links
    """

    def __init__(self, region, active_reaches=None):
        from paths import hydro_file_path

        self.id = region

        # Read hydrological input files
        self.flow_file = HydroTable(self.id, hydro_file_path, 'flow')
        self.lake_table = HydroTable(self.id, hydro_file_path, 'lake')
        self.nav = Navigator(self.id, hydro_file_path)

        # Confine to available reaches and assess what's missing
        self.active_reaches = set(active_reaches)
        self.all_reaches = self.confine
        self.active_reaches &= self.all_reaches

    @property
    def cascade(self):
        # Identify all outlets (and therefore, lakes) that exist in the current run scope
        outlets = set(self.lake_table.outlet_comid) & self.all_reaches

        # Count the number of outlets (lakes) upstream of each outlet
        reach_counts = []
        for outlet in outlets:
            upstream_lakes = len((set(self.nav.upstream_watershed(outlet)) - {outlet}) & outlets)
            reach_counts.append([outlet, upstream_lakes])
        reach_counts = pd.DataFrame(reach_counts, columns=['comid', 'n_upstream']).sort_values('n_upstream')

        # Cascade downward through tiers
        upstream_outlets = set()  # outlets from previous tier
        for tier, outlets in reach_counts.groupby('n_upstream')['comid']:
            lakes = self.lake_table[np.in1d(self.lake_table.outlet_comid, outlets)]
            all_upstream = {reach for outlet in outlets for reach in self.nav.upstream_watershed(outlet)}
            reaches = (all_upstream - set(outlets)) | upstream_outlets
            yield tier, reaches, lakes
            upstream_outlets = set(outlets)
        all_upstream = {reach for outlet in upstream_outlets for reach in self.nav.upstream_watershed(outlet)}
        yield -1, all_upstream, pd.DataFrame([])

    @property
    def confine(self):
        """ If running a series of intakes or reaches, confine analysis to upstream areas only """
        all_reaches = set(self.flow_file.index.values)
        if self.active_reaches is not None:
            all_reaches &= \
                {upstream for reach in self.active_reaches for upstream in self.nav.upstream_watershed(reach)}
        return all_reaches
