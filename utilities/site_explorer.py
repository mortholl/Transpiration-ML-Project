import pandas as pd
import os
import numpy as np

# Examines site list to pull additional relevant data for mapping and create species types breakdowns
# Writes site_locations.csv, which every later stage uses as the list of sites in the study
# Run before file_mover.py, re-running it after site_merger.py puts the merged sites back as separate rows

source_directory = r'C:\Users\thorn\Downloads\0.1.5\0.1.5\csv\sapwood'


# Group sites that sit within 1 km of each other
def group_locations(lat, lon, km=1.0):
    lat, lon = np.radians(lat), np.radians(lon)
    n = len(lat); parent = list(range(n))
    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in range(n):
        for j in range(i + 1, n):
            h = np.sin((lat[j]-lat[i])/2)**2 + np.cos(lat[i])*np.cos(lat[j])*np.sin((lon[j]-lon[i])/2)**2
            if 6371 * 2 * np.arcsin(np.sqrt(h)) < km:
                parent[root(i)] = root(j)
    return [root(i) for i in range(n)]


def name_location(sites):  # the alphabetically first site, then what distinguishes the others
    sites = sorted(sites)
    if len(sites) == 1:
        return sites[0]
    parts = [s.split('_') for s in sites]
    shared = 0
    while all(len(p) > shared and p[shared] == parts[0][shared] for p in parts):
        shared += 1
    return '_'.join([sites[0]] + ['_'.join(p[shared:]) for p in parts[1:]])

# Dropped due to insufficient data records
excluded_sites = ['SEN_SOU_POS', 'SEN_SOU_IRR', 'SEN_SOU_PRE']

site_list = pd.read_csv('data/site_list.csv')
sites = [name.split('_env')[0] for name in site_list['Unnamed: 0']]  # the sites the original study used
sites = [site for site in sites if os.path.exists(f'{source_directory}/{site}_sapf_data.csv')]  # sapwood only
sites = [site for site in sites if site not in excluded_sites]
print(f'There are {len(sites)} sites with sapwood data.')
site_files = [f'{site}_site_md.csv' for site in sites]  # named directly, matching on substrings would pick
species_files = [f'{site}_species_md.csv' for site in sites]  # up other sites whose codes start the same

site_dict = {}
for site_file in site_files:
    directory = source_directory + '/' + site_file
    md_df = pd.read_csv(directory)
    name = md_df['si_code'].values[0]
    latitude = md_df['si_lat'].values[0]
    longitude = md_df['si_long'].values[0]
    biome = md_df['si_biome'].values[0]
    map = md_df['si_map'].values[0]
    mat = md_df['si_mat'].values[0]
    site_dict.update({name: (latitude, longitude, biome, map, mat)})

site_df = pd.DataFrame.from_dict(site_dict, orient='index', columns=['Latitude', 'Longitude', 'Biome', 'MAP', 'MAT'])
groups = group_locations(site_df['Latitude'].values, site_df['Longitude'].values)
names = {g: name_location([s for s, h in zip(site_df.index, groups) if h == g]) for g in set(groups)}
site_df['Location'] = [names[g] for g in groups]  # co-located sites share one location name
site_df.to_csv('data/modeling_data/site_locations.csv', index_label='Site')

site_name_list = []
species_name_list = []
species_type_list = []
for species in species_files:
    directory = source_directory + '/' + species
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
