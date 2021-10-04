import pandas as pd
import os
import csv

site_list = pd.read_csv('data/site_list.csv')
sites = site_list.iloc[:, 0]
print(len(sites))
for site in sites:
    site = site.split('_env')[0]
loc_files = []
for filename in os.listdir('data/plant'):
    if 'site_md' in filename:
        for site in sites:
            if site[:7] in filename:
                loc_files.append(filename)
                break
print(len(loc_files))
print(loc_files)

loc_dict = {}
for location in loc_files:
    directory = 'data/plant/' + location
    df = pd.read_csv(directory)
    name = df['si_code'].values[0]
    latitude = df['si_lat'].values[0]
    longitude = df['si_long'].values[0]
    biome = df['si_biome'].values[0]
    loc_dict.update({name: (latitude, longitude, biome)})

df = pd.DataFrame.from_dict(loc_dict, orient='columns')
df.to_csv('data/site_locations.csv')