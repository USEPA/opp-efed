import arcpy
import os
from utilities import report
from paths import cdl_path, soil_raster_path, weather_path
from parameters import states_nhd, nhd_regions, vpus_nhd

overwrite = False
arcpy.CheckOutExtension("Spatial")


def clip_cdl(region, cdl_national, region_mask, years):
    report("Clipping CDL...", 1)
    for year in years:
        try:
            out_path = cdl_path.format(region, year)
            if overwrite or not os.path.exists(out_path):
                extract = arcpy.sa.ExtractByMask(cdl_national.format(year), region_mask)
                extract.save(out_path)
        except Exception as e:
            report(e)


def clip_weather(weather_raster_national, region_mask, region):
    try:
        report("Clipping weather...", 1)
        out_path = weather_path.format(region + ".tif")
        if overwrite or not os.path.exists(out_path):
            extract = arcpy.sa.ExtractByMask(weather_raster_national, region_mask)
            extract.save(out_path)
    except Exception as e:
        print(e)


def clip_soil(region, old_soils, region_mask):
    import time
    out_path = soil_raster_path.format(region)
    if overwrite or not os.path.exists(out_path):
        report("Creating soil mosaic...", 1)
        rasters = [old_soils.format(s) for s in states_nhd[region]]
        start = time.time()
        arcpy.MosaicToNewRaster_management(rasters, os.path.dirname(out_path), "temp_mosaic", number_of_bands=1)
        print(time.time() - start)
        start = time.time()
        report("Clipping soil mosaic...", 1)
        extract = arcpy.sa.ExtractByMask(os.path.join(os.path.dirname(out_path), "temp_mosaic"), region_mask)
        extract.save(out_path)
        print(time.time() - start)
        report("Cleaning up...", 1)
        arcpy.Delete_management("temp_mosaic")


def main():
    region_raster = r"T:\opp-efed\aquatic-model-inputs\bin\Input\NHDPlusV21\NHDPlus{}\NHDPlus{}\NHDPlusCatchment\cat"
    cdl_national = r"T:\NationalData\CDL\{}_30m_cdls.img"  # year
    weather_raster_national = r"T:\NationalData\weather_stations_highres_thiessens_US_alb\stationgrid2"
    old_soils = r"T:\NationalData\CustomSSURGO\{0}\{0}"

    run_settings = {'cdl': False, 'weather': False, 'soil': True}
    years = range(2013, 2018)
    for region in nhd_regions:
        if region == '07':
            print(region)
            region_mask = arcpy.Raster(region_raster.format(vpus_nhd[region], region))
            for setting, run in run_settings.items():
                if setting == 'cdl' and run:
                    clip_cdl(region, cdl_national, region_mask, years)
                if setting == 'weather' and run:
                    clip_weather(weather_raster_national, region_mask, region)
                if setting == 'soil' and run:
                    clip_soil(region, old_soils, region_mask)


main()
