import pandas as pd
import os

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

species_dict = {}
for species in species_files:
    directory = 'data/plant/' + species
    species_df = pd.read_csv(directory)
    site_name = species_df['si_code'].values[0]
    species_name = species_df['sp_name'].values[0]  # Need to update this to add all the species, not just first row
    species_type = species_df['sp_leaf_habit'].values[0]
    species_dict.update({site_name: (species_name, species_type)})

species_df = pd.DataFrame.from_dict(species_dict, orient='index', columns=['Species Name', 'Type'])
species_df.to_csv('data/species_info.csv')

species_names = species_df['Species Name'].tolist()
species_dict = {species: species_names.count(species) for species in set(species_names)}
print(species_dict)
print(f'There are {len(species_dict)} unique species in the dataset.')
species_names_df = pd.DataFrame.from_dict(species_dict, orient='index', columns=['Count'])
species_names_df.to_csv('data/species_dist.csv')


unique_species = []
species_types = {'evergreen': 0, 'deciduous': 0, 'missing': 0}
for row in species_df.iterrows():
    species = row[1][0]
    if species not in unique_species:
        unique_species.append(species)
        functional_type = str(row[1][1])
        if 'evergreen' in functional_type:
            species_types['evergreen'] += 1
        elif 'deciduous' in functional_type:
            species_types['deciduous'] += 1
        else:
            print(f'Species {species} not an evergreen or deciduous functional type')
            species_types['missing'] += 1
print(species_types)
type_df = pd.DataFrame.from_dict(species_types, orient='index', columns=['Count'])
type_df.to_csv('data/species_types.csv')