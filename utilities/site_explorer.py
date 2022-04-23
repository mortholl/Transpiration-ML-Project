import pandas as pd
import os
import numpy as np

# Examines site list to pull additional relevant data for mapping and create species types breakdowns

site_list = pd.read_csv('data/site_list.csv')
sites = site_list['Unnamed: 0'].tolist()
print(f'There are {len(sites)} sites.')
new_sites = [site.split('env')[0] for site in sites]
sites = new_sites
loc_files = []
species_files = []
for filename in os.listdir('data/plant'):
    if 'site_md' in filename:
        for site in sites:
            if site in filename:
                loc_files.append(filename)
    if 'species_md' in filename:
        for site in sites:
            if site in filename:
                species_files.append(filename)
print(f'Species files: \n {species_files}')
print(f'There are {len(species_files)} species files.')
print(f'Location files: \n {loc_files}')
print(f'There are {len(loc_files)} location files.')

loc_dict = {}
for location in loc_files:
    directory = 'data/plant/' + location
    loc_df = pd.read_csv(directory)
    name = loc_df['si_code'].values[0]
    latitude = loc_df['si_lat'].values[0]
    longitude = loc_df['si_long'].values[0]
    biome = loc_df['si_biome'].values[0]
    map = loc_df['si_map'].values[0]
    mat = loc_df['si_mat'].values[0]
    loc_dict.update({name: (latitude, longitude, biome, map, mat)})

loc_df = pd.DataFrame.from_dict(loc_dict, orient='index', columns=['Latitude', 'Longitude', 'Biome', 'MAP', 'MAT'])
loc_df.to_csv('data/modeling_data/site_locations.csv')

site_name_list = []
species_name_list = []
species_type_list = []
for species in species_files:
    directory = 'data/plant/' + species
    species_df = pd.read_csv(directory)
    for species_name in species_df['sp_name'].values:
        species_name_list.append(species_name)
        site_name_list.append(species_df['si_code'].values[0])
    for species_type in species_df['sp_leaf_habit'].values:
        species_type_list.append(species_type)


species_df = pd.DataFrame(list(zip(site_name_list, species_name_list, species_type_list)),
                          columns=['Site', 'Species Name', 'Type'],)
species_df.to_csv('data/species_info.csv', index=False)

species_names = species_df['Species Name'].tolist()
species_dict = {species: species_names.count(species) for species in set(species_names)}
print(species_dict)
print(f'There are {len(species_dict)} unique species in the dataset.')
species_names_df = pd.DataFrame.from_dict(species_dict, orient='index', columns=['Count'])
species_names_df.to_csv('data/species_dist.csv')

# Assign each site a type: evergreen, deciduous, mixed, or missing
# Count the number of each to use for a figure

type_list = []
sites = [site[0:-1] for site in sites]
types_count = {'evergreen': 0, 'deciduous': 0, 'missing': 0, 'mixed': 0}
for site in sites:
    site_types = []
    for row in species_df.iterrows():
        if row[1]['Site'] == site:
            site_types.append(row[1]['Type'])

    # Write code to remove nan from site types (this is broken)
    # for i, s_type in enumerate(site_types):
    #     if np.isnan(s_type):
    #         site_type = site_type.pop(i)

    if site == 'CRI_TAM_TOW' or site == 'SWE_NOR_ST1_BEF':  # current workaround to avoid nan
        site_type = 'missing'
        types_count['missing'] += 1
    else:
        if all([s_type == 'evergreen' for s_type in site_types]):
            site_type = 'evergreen'
            types_count['evergreen'] += 1
        elif all(['deciduous' in s_type for s_type in site_types]):
            site_type = 'deciduous'
            types_count['deciduous'] += 1
        elif all('evergreen' in s_type or 'deciduous' in s_type for s_type in site_types):
            site_type = 'mixed'
            types_count['mixed'] += 1
        else:
            site_type = 'missing'
            types_count['missing'] += 1
    type_list.append(site_type)


site_type_df = pd.DataFrame(list(zip(sites, type_list)), columns=['Site', 'Type'])
site_type_df.to_csv('data/site_types.csv')

print(types_count)
type_df = pd.DataFrame.from_dict(types_count, orient='index', columns=['Count'])
type_df.to_csv('data/species_types.csv')
