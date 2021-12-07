import os
import pandas as pd
import math

# Sandbox for data insights.

# Create .csv with LAI when available
location_list = []
lai_list = []
for filename in os.listdir('data/plant'):
    if "stand_md" in filename:
        location = filename.split('_stand')[0]
        stand_df = pd.read_csv('data/plant/'+filename)
        lai = float(stand_df['st_lai'])
        if math.isnan(lai) == False:
            location_list.append(location)
            lai_list.append(lai)

df = pd.DataFrame({'Location': location_list, 'LAI': lai_list})
df = df.set_index(['Location'])
df.to_csv('data/lai.csv')
