import os

# Root directories
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sam_root = os.path.join(project_root, "..", "bin")
table_root = os.path.join(sam_root, "Tables")
input_root = os.path.join("..", "..", "aquatic-model-inputs", "bin", "Production")

# Input data
weather_path = os.path.join(sam_root, "WeatherArray")
hydro_file_path = os.path.join(sam_root, "HydroFiles", "region_{}_{}.npz")  # region, type
recipe_path = os.path.join(input_root, "RecipeFiles", "r{}_{}.npz")  # region, year
stage_one_scenario_path = os.path.join(input_root, "SamScenarios", "r{}.csv")  # region, year
stage_two_scenario_path = os.path.join(sam_root, "StageTwoScenarios", "r{}")  # region, year

# Tables
endpoint_format_path = os.path.join(table_root, "endpoint_format.csv")
fields_and_qc_path = os.path.join(table_root, "fields_and_qc.csv")
types_path = os.path.join(table_root, "tr_55.csv")

# Drinking water intakes paths
dwi_path = os.path.join(sam_root, "Intakes", "intake_locations.csv")
manual_points_path = os.path.join(sam_root, "Intakes", "mtb_intakes.csv")

# SAM run paths

output_path = os.path.join(sam_root, "Results")
