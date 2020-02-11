import os
import arcpy
import numpy as np
import pandas as pd

from utilities import report
from parameters import vpus_nhd, nhd_regions
import time

from paths import soil_raster_path, weather_path, cdl_path, combo_path, combined_raster_path


def overlay_rasters(outfile, *rasters):
    for raster in map(arcpy.Raster, rasters):
        if not raster.hasRAT:
            report("Building RAT for {}".format(raster.catalogPath), 1)
            arcpy.BuildRasterAttributeTable_management(raster)
    arcpy.gp.Combine_sa([arcpy.Raster(r) for r in rasters], outfile)


def generate_combos(combined_raster, combinations_table):
    # new field name, beginning of old field name
    field_map = [('combo_id', 'VALUE'),
                 ('count', 'COUNT'),
                 ('mukey', 'SOIL'),
                 ('cdl', 'CDL'),
                 ('weather_grid', 'MET'),
                 ('gridcode', 'CAT')]
    raw_fields = [f.name for f in arcpy.ListFields(combined_raster)]
    field_dict = {}
    for new_name, search in field_map:
        for old_name in raw_fields:
            if old_name.startswith(search):
                field_dict[old_name] = new_name
                break
        else:
            field_dict[old_name] = None
    if all(field_dict.values()):
        old_fields, new_fields = zip(*sorted(field_dict.items()))
        data = np.array([row for row in arcpy.da.SearchCursor(combined_raster, old_fields)])
        table = pd.DataFrame(data, columns=new_fields)
        table['area'] = table['count'] * 900
        table.to_csv(combinations_table, index=None)
    else:
        raise KeyError("Missing fields in attribute table")


def main():
    years = range(2013, 2018)  # range(2010, 2016)
    arcpy.CheckOutExtension("Spatial")
    arcpy.env.overwriteOutput = True
    overwrite_rasters = False
    overwrite_tables = True

    # Create output directory
    if not os.path.exists(os.path.dirname(combo_path)):
        os.makedirs(os.path.dirname(combo_path))

        # Iterate through year/region combinations
    regions = nhd_regions
    for region in regions:
        soil_raster = soil_raster_path.format(region)
        weather_raster = weather_path.format(region) + ".tif"
        for year in years:
            cdl_raster = cdl_path.format(region, year)
            combined_raster = combined_raster_path.format(region, year)
            combinations_table = combo_path.format(region, year)
            if overwrite_rasters or not os.path.exists(combined_raster):
                try:
                    if all(map(os.path.exists, (soil_raster, weather_raster, cdl_raster))):
                        report("Performing raster overlay for Region {}, {}...".format(region, year))
                        overlay_rasters(combined_raster, soil_raster, cdl_raster, weather_raster, nhd_raster)
                    else:
                        paths = [('soil', soil_raster),
                                 ('weather', weather_raster), ('cdl', cdl_raster)]
                        missing = ", ".join([name for name, path in paths if not os.path.exists(path)])
                        report("Missing {} layers for Region {}, {}".format(missing, region, year))
                except Exception as e:
                    print(e)
            if overwrite_tables or not os.path.exists(combinations_table):
                report("Generating combination table for Region {}, {}...".format(region, year))
                start = time.time()
                try:
                    generate_combos(combined_raster, combinations_table)
                except Exception as e:
                    print(e)
                print(time.time() - start)


main()
