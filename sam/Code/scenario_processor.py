import dask
import numpy as np
from model_functions import stage_one_to_two
from classes import WeatherArray, StageOneScenarios, StageTwoScenarios


# TODO - irrigation should go in precip?
# TODO - handling of non-ag classes
# TODO - null soil values - why?
# TODO - missing crop dates?
# TODO - are fill values working?


def main():
    regions = ['07']

    # Initialize input met matrix
    met = WeatherArray()

    for region in regions:
        report("Processing Region {} scenarios...".format(region))

        # Initialize output
        report("Generating stage 2 scenarios...", 1)
        stage_two = StageTwoScenarios(region, met)

        # Run weather simulations to generate stage two scenarios
        stage_two.build_from_stage_one()

if __name__ == "__main__":
    # import cProfile

    # cProfile.run('main()')
    main()
