import os
import boto3
import botocore
import re
import zipfile
import time

from utilities import report
from paths import nhd_path, soil_path, weather_path, cdl_path, staged_dir, table_dir, shapefile_path
from paths import remote_nhd_path, remote_soil_path, remote_weather_path, remote_cdl_path, remote_table_path, \
    remote_shapefile_path

session = boto3.session.Session(profile_name='sam')
s3 = session.resource('s3')
sam_input_bucket = s3.Bucket('sam-raw-inputs')


def download_file(remote, local):
    if not os.path.exists(os.path.dirname(local)):
        os.makedirs(os.path.dirname(local))
    report("Downloading file {}...".format(remote), 1)
    sam_input_bucket.download_file(remote, local)


def download_batch(remote_files, local_files=None, retain_zip=False):
    if local_files is None:
        local_files = [None for _ in remote_files]
    for remote_file, local_file in zip(remote_files, local_files):

        try:
            if local_file is None or overwrite_local or not os.path.exists(local_file):
                local_zip = os.path.join(staged_dir, os.path.basename(remote_file))
                if not os.path.exists(local_zip):
                    report("Downloading file {}...".format(remote_file), 1)
                    sam_input_bucket.download_file(remote_file, local_zip)
                if local_file is not None:
                    report("Extracting...", 2)
                    with zipfile.ZipFile(local_zip) as zf:
                        zf.extractall(os.path.dirname(local_file))
                if not retain_zip:
                    os.remove(local_zip)
        except Exception as e:
            report(e)


def acquire_nhd(vpu, region):
    remote_file_format = re.compile("/NHDPlusV21_[A-Z]{2}_[A-Za-z0-9]{2,3}_(\D+?)_")
    remote_dir = remote_nhd_path.format(vpu, region)
    remote_files = [obj.key for obj in sam_input_bucket.objects.filter(Prefix=remote_dir)]
    # TODO - figure out how to unzip 7z files
    #local_files = \
    #    [os.path.join(nhd_path.format(vpu, region), re.search(remote_file_format, f).group(1)) for f in remote_files]

    download_batch(remote_files, retain_zip=True)


def acquire_soils(states):
    remote_files = [remote_soil_path.format(state.lower()) for state in states]
    local_files = [soil_path.format(state) for state in states]
    download_batch(remote_files, local_files)


def acquire_cdl(region, cdl_years):
    remote_files = [remote_cdl_path.format(region, year) for year in cdl_years]
    local_files = [cdl_path.format(year, region) for year in cdl_years]
    download_batch(remote_files, local_files)


def acquire_weather(region):
    remote_file = remote_weather_path.format(region)
    local_file = weather_path.format(region)
    download_batch([remote_file], [local_file])


def acquire_tables():
    remote_files = [obj.key for obj in sam_input_bucket.objects.filter(Prefix=remote_table_path)]
    local_files = [os.path.join(table_dir, os.path.basename(f)) for f in remote_files]
    for remote, local in zip(remote_files, local_files):
        try:
            download_file(remote, local)
        except PermissionError:
            pass


def main():
    from parameters import states_nhd, vpus_nhd, nhd_regions
    global overwrite_local
    overwrite_local = False
    #nhd_regions = ['07']
    cdl_years = range(2013, 2018)
    for region in nhd_regions:
        vpu = vpus_nhd[region]
        states = states_nhd[region]

        # Download and unzip files
        # acquire_nhd(vpu, region)
        # acquire_cdl(region, cdl_years)
        # acquire_weather(region)
        # acquire_soils(states)

    acquire_tables()


main()
