## Import packages
import numpy as np
import pandas as pd

## Import data and sanitize ##

# Import data
# Meteorological data
def sanitize_weather(row):
    if all([row[0], row[1], row[2], row[3], row[4]]):  # can insert more data sanitizing conditions here
        return {str(row[0]):  # Time stamp for data matching
                    [float(row[1]),  # Solar radiation, W/m^2
                     float(row[2]),  # Air temp, deg C
                     float(row[3]),  # Relative humidity, %
                     float(row[4])]}  # Vapor pressure deficit, kPa
    else:
        return False


weather = {}
import csv

with open('data/weather_30min.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        data = sanitize_weather(row)
        if data:
            weather.update(data)

# Rainfall data
rainfall = {}
import csv

with open('data/daily_rain.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        if row:
            rainfall.update({str(row[0]): float(row[1])})  # daily rainfall in mm

# Import soil data here using global averages between all depths, all locations
soil_xls = pd.ExcelFile('data/Plantation soil moisture.xls')
soil_data = pd.read_excel(soil_xls, 'Averages')
soil_keys = soil_data['Date Time'].to_list()
soil_values = soil_data['Global'].to_list()
soil_dict = {}
for s, date in enumerate(soil_keys):
    soil_dict.update({str(date): float(soil_values[s])})

# Add groundwater level data
# Use 3663 for 17-18
GWL_xls = pd.ExcelFile('data/groundwater level.xls')
gwl = pd.read_excel(GWL_xls, '3663')
gwl_keys = gwl['datetime'].to_list()
gwl_values = gwl['waterlevel_corr'].to_list()
gwl_1718 = {}
for h, date in enumerate(gwl_keys):
    gwl_1718.update({str(date): float(gwl_values[h])})

# Use 3666 for 19-20
GWL_xls = pd.ExcelFile('data/groundwater level.xls')
gwl = pd.read_excel(GWL_xls, '3666')
gwl_keys = gwl['datetime'].to_list()
gwl_values = gwl['waterlevel_corr'].to_list()
gwl_1920 = {}
for h, date in enumerate(gwl_keys):
    gwl_1920.update({str(date): float(gwl_values[h])})

gwl_full = {**gwl_1718, **gwl_1920}

# Transpiration data (based on sap flux measurements)
transpiration_1718 = []
transpiration_1920 = []
import csv

with open('data/avg_transp.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        if row[1]:
            transpiration_1718.append([str(row[0]), float(row[1])])
        if row[2]:
            transpiration_1920.append([str(row[0]), float(row[2])])

transpiration = transpiration_1718 + transpiration_1920

# Store available data into X and Y matrices
X = []
Y = []
timestamp_full = []
for row in transpiration:
    date = row[0]
    x = []
    x1 = weather.get(date,'missing') #search weather dictionary for time stamp - if missing return False
    x2 = rainfall.get(date[:10],'missing') #same for rainfall dictionary
    x3 = gwl_full.get(date,'missing')
    if x1 != 'missing' and x2 != 'missing' and x3 != 'missing':
        x.extend(x1)
        x.append(x2)
        x.append(x3)
        X.append(x)
        Y.append(row[1])
        timestamp_full.append(date)


# "X": R [W/m^2], Temp [deg C], RH [%], VPD [kPa], rainfall [mm/d]
# "Y": target transpiration [mm/d]
X = np.array(X)
Y = np.array(Y) * 86400 # conversion factor to [mm/d]

# Create X and Y matrices for separating '17-'18 data
X_1718 = []
Y_1718 = []
timestamp_1718 = []
for row in transpiration_1718:
    date = row[0]
    x = []
    x1 = weather.get(date,'missing') #search weather dictionary for time stamp - if missing return False
    x2 = rainfall.get(date[:10],'missing') #same for rainfall dictionary
    x3 = gwl_1718.get(date,'missing')
    if x1 != 'missing' and x2 != 'missing' and x3 != 'missing':
        x.extend(x1)
        x.append(x2)
        x.append(x3)
        X_1718.append(x)
        Y_1718.append(row[1])
        timestamp_1718.append(date)

# "X": R [W/m^2], Temp [deg C], RH [%], VPD [kPa], rainfall [mm/d]
# "Y": target transpiration [mm/d]
X_1718 = np.array(X_1718)
Y_1718 = np.array(Y_1718) * 86400 # conversion factor to [mm/d]

# Create X and Y matrices for separating '19-'20 data
X_1920 = []
Y_1920 = []
timestamp_1920 = []
for row in transpiration_1920:
    date = row[0]
    x = []
    x1 = weather.get(date,'missing') #search weather dictionary for time stamp - if missing return False
    x2 = rainfall.get(date[:10],'missing') #same for rainfall dictionary
    x3 = soil_dict.get(date,'missing') #add in soil moisture for '19-'20 data
    #x4 = gwl_1920.get(date,'missing')
    if x1 != 'missing' and x2 != 'missing' and x3 != 'missing':
        x.extend(x1)
        x.append(x2)
        x.append(x3)
        #x.append(x4)
        X_1920.append(x)
        Y_1920.append(row[1])
        timestamp_1920.append(date)

# "X": R [W/m^2], Temp [deg C], RH [%], VPD [kPa], rainfall [mm/d], soil moisture [units?]
# "Y": target transpiration [mm/d]
X_1920 = np.array(X_1920)
Y_1920 = np.array(Y_1920) * 86400 # conversion factor to [mm/d]

print(f"Mean transpiration for '19-'20 data is {round(np.mean(Y_1920),4)} mm/d")
print(f"Mean transpiration for '17-'18 data is {round(np.mean(Y_1718),4)} mm/d")
print(f"Mean overall transpiration is {round(np.mean(Y),4)} mm/d")
print('\nStandard deviations:')
print(np.std(Y_1920))
print(np.std(Y_1718))
print(np.std(Y))

## Standardize and split data ##

#Set aside testing and validation sets: 80% training, 10% testing, 10% validation
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size = 0.5, random_state = 42)

#Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)
X = scaler.transform(X)

#Define m and n using input features
m = len(Y_train)
n = len(X_train[0]) +1
print("m = ", m, "for combined data")
print("n = ", n)

#Repeat for separate '17-'18 and '19-'20 data
from sklearn.model_selection import train_test_split
X_train_1718, X_test_1718, Y_train_1718, Y_test_1718 = train_test_split(X_1718, Y_1718, test_size = 0.2, random_state = 42)
X_test_1718, X_val_1718, Y_test_1718, Y_val_1718 = train_test_split(X_test_1718, Y_test_1718, test_size = 0.5, random_state = 42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_1718 = scaler.fit_transform(X_train_1718)
X_test_1718 = scaler.transform(X_test_1718)
X_val_1718 = scaler.transform(X_val_1718)
X_1718 = scaler.transform(X_1718)

m = len(Y_train_1718)
n = len(X_train_1718[0]) +1
print("m = ", m, "for '17-'18 data")

from sklearn.model_selection import train_test_split
X_train_1920, X_test_1920, Y_train_1920, Y_test_1920 = train_test_split(X_1920, Y_1920, test_size = 0.2, random_state = 42)
X_test_1920, X_val_1920, Y_test_1920, Y_val_1920 = train_test_split(X_test_1920, Y_test_1920, test_size = 0.5, random_state = 42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_1920 = scaler.fit_transform(X_train_1920)
X_test_1920 = scaler.transform(X_test_1920)
X_val_1920 = scaler.transform(X_val_1920)
X_1920 = scaler.transform(X_1920)

m = len(Y_train_1920)
n = len(X_train_1920[0]) +1
print("m = ", m, "for '19-'20 data")