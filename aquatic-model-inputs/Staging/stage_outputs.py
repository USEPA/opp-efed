import os
import boto3
import botocore
import re
import zipfile
import time

from utilities import report
from paths import weather_path, hydro_file_path, recipe_path, scenario_matrix_path
from paths import remote_metfile_path, remote_hydrofile_path, remote_recipe_path, remote_scenario_path

session = boto3.session.Session(profile_name='sam')
s3 = session.resource('s3')
sam_staged_bucket = s3.Bucket('sam-staged-inputs')


def upload_hydro(region):
    for table in ("lake", "flow", "nav"):
        local = hydro_file_path.format(region).format(table)
        remote = os.altsep.join((remote_hydrofile_path, os.path.basename(local)))
        upload_file(local, remote)


def upload_scenarios(region):
    local = scenario_matrix_path.format('sam_aggregated', region)
    remote = remote_scenario_path.format(region)
    upload_file(local, remote)


def upload_recipes(region, years):
    for year in years:
        local = recipe_path.format(region, year)
        remote = remote_recipe_path.format(region, year)
        upload_file(local, remote)


def upload_weather(region):
    local_root = weather_path.format(region)
    remote_root = remote_metfile_path.format(region)
    for item in ('weather_cube.dat', 'weather_grid.csv', 'key.csv'):
        local = os.path.join(local_root, item)
        remote = os.altsep.join((remote_root, os.path.basename(local)))
        upload_file(local, remote)


def upload_file(local, remote):
    report("{} -> {}".format(local, remote))
    try:
        sam_staged_bucket.upload_file(local, remote)
    except Exception as e:
        print(e)


def main():
    from parameters import nhd_regions
    nhd_regions = ['07']
    cdl_years = range(2013, 2018)

    for region in nhd_regions:
        #upload_hydro(region)
        #upload_scenarios(region)
        #upload_recipes(region, cdl_years)
        upload_weather(region)


main()
