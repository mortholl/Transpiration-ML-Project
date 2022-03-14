import pandas as pd
import os

# Examines data and comes up with a site list based on available environmental data

directory = 'data/plant'
column_dict = {}
column_count = {}
for filename in os.listdir(directory):                 # for each file in 'data/plant' get the file name
    if 'env_data' in filename:                         # if the file name contains env_data
        path = os.path.join(directory, filename)
        file = pd.read_csv(path)
        column_dict.update({filename: file.columns})   # create dictionary with the file name : column heading values
        for col_name in file.columns:                  # keep a running count of sites for each column heading
            if col_name in column_count:
                column_count[col_name] += 1
            else:
                column_count.update({col_name: 1})

print(column_count)
print(f'The total number of sites is {len(column_dict)}.')


def new_dicts(new_site_list, old_column_dict):
    new_column_count = {}
    new_column_dict = {}
    for new_site in new_site_list:
        values = old_column_dict[new_site]
        new_column_dict.update({new_site: values})
        for col_name in values:
            if col_name in new_column_count:
                new_column_count[col_name] += 1
            else:
                new_column_count.update({col_name: 1})
    return new_column_count, new_column_dict


site_list = []
for data_site, variables in column_dict.items():
    if 'swc_shallow' in variables:
        site_list.append(data_site)
column_count, column_dict = new_dicts(site_list, column_dict)
print(column_count)
print(f'Total number of sites with shallow soil water content is {len(site_list)}.')

site_list = []
for data_site, variables in column_dict.items():
    if all(value in variables for value in ['precip', 'ta', 'vpd', 'rh', 'ppfd_in']):
        site_list.append(data_site)
column_count, column_dict = new_dicts(site_list, column_dict)
print(column_count)
print(f'Total number of sites is {len(site_list)}.')

directory = 'data/leaf'
count = 0
leaf_sites = []
for site in site_list:
    site = site.split('_env')[0]
    for filename in os.listdir(directory):
        if site in filename:
            leaf_sites.append(filename)
            count += 1
            break
print(f'\n {leaf_sites}')
print(f'The number of sites that can be correlated with leaf data is {count}.')


df = pd.DataFrame.from_dict(column_dict, orient='index')
df.to_csv('data/site_list.csv')
