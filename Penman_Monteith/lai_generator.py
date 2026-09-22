import os
import pandas as pd
import math

# Create .csv with LAI when available
site_list = []
lai_list = []
for filename in os.listdir('data/plant'):
    if "stand_md" in filename:
        site = filename.split('_stand')[0]
        stand_df = pd.read_csv('data/plant/'+filename)
        lai = float(stand_df['st_lai'])
        if math.isnan(lai) == False:
            site_list.append(site)
            lai_list.append(lai)

df = pd.DataFrame({'Site': site_list, 'LAI': lai_list})
df = df.set_index(['Site'])
df.to_csv('data/lai.csv')
