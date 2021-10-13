import pandas as pd
import os
import shutil


# Moves all relevant data into modeling_data folder

site_location_df = pd.read_csv('data/modeling_data/site_locations.csv')
sites = site_location_df['Unnamed: 0'].tolist()

target_files = []
training_files = []
for filename in os.listdir('data/plant'):
    if 'sapf_data' in filename:
        for site in sites:
            if site in filename:
                target_files.append(filename)
    if 'env_data' in filename:
        for site in sites:
            if site in filename:
                training_files.append(filename)

print(len(target_files))
print(len(training_files))

# Save training and target files to modeling data
for filename in target_files:
    source_directory = 'data/plant/'+filename
    destination_directory = 'data/modeling_data/targets/'+filename
    shutil.copyfile(source_directory, destination_directory)
for filename in training_files:
    source_directory = 'data/plant/'+filename
    destination_directory = 'data/modeling_data/features/'+filename
    shutil.copyfile(source_directory, destination_directory)
