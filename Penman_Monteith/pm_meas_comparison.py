import statsmodels.api as sm
import pandas as pd
import os
from math import isnan
import matplotlib.pyplot as plt

# File for comparing the Penman-Monteith predictions with the measured transpiration values.
# Plot values against each other, and also plot values over time for selected sites.
# Linear regression to see how well they correlate - potentially come up with equation to describe relationship?

PM = []  # list of Penman-Monteith predictions
SF = []  # list of measured sap flux values

for filename in os.listdir('Penman_Monteith/sap_flux_um-sec'):
    location = filename.split('_um')[0]
    sap_flux_df = pd.read_csv('Penman_Monteith/sap_flux_um-sec/'+location+'_um-sec.csv')
    pm_df = pd.read_csv('Penman_Monteith/pm_prediction/'+location+'_pm_transpiration.csv')
    pm_series = pd.Series(pm_df['Reference ET'])
    sf_series = pd.Series(sap_flux_df['Average Sap Flux'])
    if len(pm_series) != len(sf_series):
        print(f"Number of PM predictions doesn't match number of SF measurements for {location}.")

    for i, sf in enumerate(sf_series):
        if not isnan(sf):  # check that the sap flux value is a number
            pm = pm_series[i]
            if not isnan(pm):  # check that PM is a number
                PM.append(pm)
                SF.append(sf)
    print(f'{location} done')

lin_reg = sm.OLS(PM, SF).fit()
print(lin_reg.summary())
plt.scatter(PM, SF)
plt.show()

# plot time series - maybe multiply PM by 100 to get on same scale
