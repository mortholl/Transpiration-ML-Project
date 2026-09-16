import pandas as pd
import os
import shutil


# Moves all relevant data into modeling_data folder

source_directory = r'C:\Users\thorn\Downloads\0.1.5\0.1.5\csv\sapwood'
feature_directory = 'data/modeling_data/features'
target_directory = 'data/modeling_data/targets'

site_location_df = pd.read_csv('data/modeling_data/site_locations.csv')
sites = site_location_df['Unnamed: 0'].tolist()

target_files = []
training_files = []
for filename in os.listdir(source_directory):
    if 'sapf_data' in filename:
        for site in sites:
            if site in filename:
                target_files.append(filename)
    if 'env_data' in filename:
        for site in sites:
            if site in filename:
                training_files.append(filename)

# Save training and target files to modeling data
for filename in target_files:
    shutil.copyfile(os.path.join(source_directory, filename), os.path.join(target_directory, filename))
for filename in training_files:
    shutil.copyfile(os.path.join(source_directory, filename), os.path.join(feature_directory, filename))
