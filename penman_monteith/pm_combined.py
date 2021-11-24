from Penman_Monteith import equations
import os
import pandas as pd
import datetime

# Generates Penman-Monteith evapotranspiration predictions based on environmental data

begin_time = datetime.datetime.now()

# Radiation [W/m2], Air Temperature [K], Specific Humidity [kg/kg], Wind Speed [m/s]
# Reference Evapotranspiration [um/sec]
feature_directory = 'data/modeling_data/features'
features = ['TIMESTAMP', 'ppfd_in', 'ta', 'rh']
wind_speed_df = pd.read_csv('data/modeling_data/avg_wind_speed.csv')
for filename in os.listdir(feature_directory):
    df = pd.read_csv(feature_directory + '/' + filename, header=0, sep=',', usecols=features)
    location = filename.split('_env')[0]
    new_filename = location + '_pm_transpiration'
    monthly_wind_speed = wind_speed_df[wind_speed_df['Site Name'] == location]
    for i, row in df.iterrows():
        if all(row) != 'NA':
            month = row['TIMESTAMP'][5:7]
            month = month.strip('0')  # Remove leading zeros from timestamp month
            u = float(monthly_wind_speed[month])  # Average wind speed for the location in that month
            phi = row['ppfd_in']*0.43  # Conversion factor from ppfd_in to net radiation
            ta = row['ta']+273  # Convert deg C to K
            qa = equations.qaRh(row['rh'], ta)  # Use equation to calculate specific humidity
            ref_ET = equations.evfPen(phi, qa, ta, u)  # Calculate reference ET (mm/day) using Penman-Monteith equation
            df.at[i, 'Reference ET'] = ref_ET

    df.to_csv('Penman_Monteith/pm_prediction/'+new_filename+'.csv')
    print(f'{new_filename} finished')

end_time = datetime.datetime.now()
print(f'The runtime was {end_time - begin_time}.')
