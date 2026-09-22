import pandas as pd
import os
import shutil


# Moves all relevant data into modeling_data folder

source_directory = r'C:\Users\thorn\Downloads\0.1.5\0.1.5\csv\sapwood'
feature_directory = 'data/modeling_data/features'
target_directory = 'data/modeling_data/targets'

sites = pd.read_csv('data/modeling_data/site_locations.csv')['Site']  # named directly, matching on
target_files = [f'{site}_sapf_data.csv' for site in sites]    # substrings would pick up a merged name as
training_files = [f'{site}_env_data.csv' for site in sites]   # a prefix of the sites it was built from


def copy_file(filename, destination):  # Copies one file, skipping anything the download does not have
    source = os.path.join(source_directory, filename)
    if not os.path.exists(source):
        print(f'{filename} is not in the download, skipped')
        return
    shutil.copyfile(source, os.path.join(destination, filename))


# Save training and target files to modeling data
for name in target_files:
    copy_file(name, target_directory)
for name in training_files:
    copy_file(name, feature_directory)
