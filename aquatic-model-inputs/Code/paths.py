import os

# Root directories
#root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "bin"))
root_dir = r"J:\opp-efed-data\aquatic-model-inputs\bin"
table_dir = os.path.join(root_dir, "Tables")
#input_dir = os.path.join(root_dir, "Input")  # region
input_dir = r"J:\NationalData"
intermediate_dir = os.path.join(root_dir, "Intermediate")
production_dir = os.path.join(root_dir, "Production")
staged_dir = os.path.join(root_dir, "Staged")
scratch_dir = os.path.join(root_dir, "Scratch")

# Raw input data
nhd_path = os.path.join(input_dir, "NHDPlusV21", "NHDPlus{}", "NHDPlus{}")  # vpu, region
soil_path = os.path.join(input_dir, "SSURGO", "gSSURGO_{}.gdb")  # state
condensed_soil_path_old = os.path.join(input_dir, "CustomSSURGO")
condensed_soil_path = os.path.join(input_dir, "CustomSSURGO", "{}", "{}.csv")  # state, table name
cdl_path = os.path.join(input_dir, "CDL", "cdl{}_{}.img")  # region, year
weather_path = os.path.join(input_dir, "WeatherFiles", "met{}")  # region

# Rasters
nhd_raster_path = os.path.join(nhd_path, "NHDPlusCatchment", "cat")
# soil_raster_path = os.path.join(soil_path, "MapunitRaster_10m")
soil_raster_path = os.path.join(input_dir, "SoilMap", "soil{}.tif")  # region

# Intermediate datasets
combo_path = os.path.join(intermediate_dir, "Combinations", "{}_{}.csv")  # region, state, year
met_grid_path = os.path.join(intermediate_dir, "Weather", "met_stations.csv")
processed_soil_path = os.path.join(intermediate_dir, "ProcessedSoils", "{}", "region_{}")  # mode, region
combined_raster_path = os.path.join(intermediate_dir, "CombinedRasters", "c{}_{}")

# Table paths
crop_params_path = os.path.join(table_dir, "cdl_params.csv")
gen_params_path = os.path.join(table_dir, "curve_numbers.csv")
crop_dates_path = os.path.join(table_dir, "crop_dates.csv")
crop_group_path = os.path.join(table_dir, "crop_groups.csv")
met_attributes_path = os.path.join(table_dir, "met_params.csv")
fields_and_qc_path = os.path.join(table_dir, "fields_and_qc.csv")
irrigation_path = os.path.join(table_dir, "irrigation.csv")
volume_path = os.path.join(table_dir, "lake_morpho.csv")

# Misc paths
shapefile_path = os.path.join(scratch_dir, "Shapefiles")
remote_shapefile_path = os.altsep.join(("National", "Shapefiles"))

# Production data
hydro_file_path = os.path.join(production_dir, "HydroFiles", "region_{}_{{}}.npz")  # region, type
recipe_path = os.path.join(production_dir, "RecipeFiles", "r{}_{}.npz")  # region, year
sam_scenario_path = os.path.join(production_dir, "SamScenarios", "r{}.csv")  # region, year
pwc_scenario_path = os.path.join(production_dir, "PwcScenarios", "{1}", "r{0}_{1}.csv")  # region, crop name
pwc_metfile_path = os.path.join(production_dir, "PwcMetfiles", "s{}.csv")

# Remote input data
remote_nhd_path = os.altsep.join(("NHD", "NHDPlus{}", "NHDPlus{}"))  # vpu, region
remote_cdl_path = os.altsep.join(("CDL", "r{}_{}.zip"))  # region, year
remote_soil_path = os.altsep.join(("SSURGO", "gssurgo_g_{}.zip"))  # state
remote_weather_path = os.altsep.join(("Weather", "region{}.zip"))  # region
remote_table_path = os.altsep.join(("Parameters",))

# Remote production data
remote_metfile_path = os.altsep.join(("WeatherArray", "region{}"))
remote_hydrofile_path = os.altsep.join(("HydroFiles",))
remote_recipe_path = os.altsep.join(("Recipes", "region_{}_{}.npz"))
remote_scenario_path = os.altsep.join(("Scenarios", "region_{}.csv"))
