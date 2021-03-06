import pandas as pd
import os
import datetime

# Calculates average sap flux data in desired units

# begin_time = datetime.datetime.now()

for filename in os.listdir('data/modeling_data/targets'):
    location = filename.split('_sapf')[0]
    # df = pd.read_csv('data/modeling_data/targets/' + filename)
    # df['Average Sap Flux'] = df.iloc[:, 2:].mean(axis=1)  # takes average excluding time stamps
    # df[['TIMESTAMP', 'Average Sap Flux']].to_csv('Penman_Monteith/sap_flux_cm3-sec/'+location+'_cm3-sec.csv')  # Saves sap flux in cm3/s for all locations

    for filename_leaf in os.listdir('data/leaf'):  # Saves the subset of available leaf data, converted to um/sec
        if location in filename_leaf and 'sapf_data' in filename_leaf:
            leaf_df = pd.read_csv('data/leaf/'+location+'_sapf_data.csv')
            leaf_df['Average Sap Flux'] = leaf_df.iloc[:, 2:].mean(axis=1) * 10000 / 3600  # conversion factor included from cm/hr to um/s
            leaf_df[['TIMESTAMP', 'Average Sap Flux']].to_csv('Penman_Monteith/sap_flux_um-sec/'+location+'_um-sec.csv')
            break  # breaks the file finding loop after the csv is created
    print(f'{location} complete')

# end_time = datetime.datetime.now()
# print(f'The runtime was {end_time - begin_time}.')
