import pandas as pd
import os


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
print(f'The total number of sites is {len(column_dict)}')


def new_dicts(site_list, old_column_dict):
    column_count = {}
    column_dict = {}
    for site in site_list:
        values = old_column_dict[site]
        column_dict.update({site: values})
        for col_name in values:
            if col_name in column_count:
                column_count[col_name] += 1
            else:
                column_count.update({col_name: 1})
    return column_count, column_dict


site_list = []
for site, values in column_dict.items():
    if 'swc_shallow' in values:
        site_list.append(site)
column_count, column_dict = new_dicts(site_list, column_dict)
print(column_count)
print(f'Total number of sites with shallow soil water content is {len(site_list)}')


column_count, column_dict = new_dicts(site_list, column_dict)
for site in site_list:
    if 'precip' not in column_dict[site]:
        if site in site_list:
            site_list.remove(site)
column_count, column_dict = new_dicts(site_list, column_dict)
print(column_count)
print(f'Total number of sites with precipitation is {len(site_list)}')

directory = 'data/leaf'
count = 0
leaf_sites = []
for site in site_list:
    for filename in os.listdir(directory):
        if site[:7] in filename:
            leaf_sites.append(filename)
            count += 1
            break
print(f'\n {leaf_sites}')
print(f'The number of sites that can be correlated with leaf data is {count}')


df = pd.DataFrame.from_dict(column_dict, orient='index')
df.to_csv('data/site_list.csv')
