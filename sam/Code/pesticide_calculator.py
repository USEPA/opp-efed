import os
from utilities import report
from classes import InputParams, WatershedHydrology, StageThreeScenarios, Outputs, Recipes, Reaches


def initialize():
    # Make sure needed subdirectories exist
    preexisting_subdirs = (os.path.join("..", "bin", d) for d in ("Results", "temp"))
    for subdir in preexisting_subdirs:
        if not os.path.exists(subdir):
            os.makedirs(subdir)

    # Purge temp folder
    temp_folder = os.path.join("..", "bin", "temp")
    for f in os.listdir(temp_folder):
        os.remove(os.path.join(temp_folder, f))


def pesticide_calculator(input_data):
    # Initialize file structure
    initialize()

    # Initialize parameters from ront end
    inputs = InputParams(input_data)

    # Loop through all NHD regions included in selected runs
    for region_id in inputs.active_regions:
        report("Processing hydroregion {}...".format(region_id))

        # Load watershed topology maps and account for necessary files
        region = WatershedHydrology(region_id, inputs.active_reaches)

        # Simulate application of pesticide to all input scenarios
        scenarios = StageThreeScenarios(region_id, inputs.crops)  # retain='une'

        # Initialize output object
        outputs = Outputs(inputs, scenarios.names, scenarios.start_date, scenarios.end_date)

        # Cascade downstream processing watershed recipes and performing travel time analysis
        for year in [2013]:  # manual years

            # Load recipes for region and year
            recipes = Recipes(region_id, year)

            # Combine scenarios to generate data for catchments
            report("Processing recipes for {}...".format(year))
            reaches = Reaches(scenarios, recipes, region, outputs)

            # Traverse downstream in the watershed
            for tier, reach_ids, lakes in region.cascade:

                print(tier, len(reach_ids))

                # Confine reaches to those not already run
                reach_ids -= reaches.burned_reaches

                # Calculate runoff and mass time series for all reaches
                for reach_id in reach_ids:
                    reaches.process_local(reach_id)

                # Perform full analysis including time-of-travel and concentration for active reaches
                for reach_id in reach_ids & region.active_reaches:
                    reaches.process_upstream(reach_id)

                # Pass each reach in the tier through a downstream lake
                for _, lake in lakes.iterrows():
                    reaches.burn_lake(lake)

                reaches.burned_reaches |= reach_ids

        # Write output
        report("Writing output...")
        outputs.write_output()


if __name__ == "__main__":
    from Development.test_inputs import atrazine_json
    from sam_exe_dev import Sam

    input_dict = Sam(atrazine_json).input_dict

    if False:
        cProfile.run('PesticideCalculator(input_dict)')
    else:
        pesticide_calculator(input_dict)
