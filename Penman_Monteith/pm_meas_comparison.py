import statsmodels.api as sm
import pandas as pd
import os
from math import isnan
import numpy as np
import matplotlib.pyplot as plt

# File for comparing the Penman-Monteith predictions with the measured transpiration values.
# Plot values against each other, and also plot values over time for selected sites.
# Linear regression to see how well they correlate and describe correlation.


class PMsfCompare:
    def __init__(self):
        self.PM = []  # list of Penman-Monteith predictions
        self.SF = []  # list of measured sap flux values
        self.PM_env = []  # list of Penman Monteith predictions to use for comparison with environmental variables
        self.SF_env = []  # list of sap flux measurements to use for comparison with environmental variables
        self.TA = []  # list of air temperature values, deg C
        self.RH = []  # list of relative humidity values, %
        self.VPD = []  # list of vapor pressure deficit values
        self.PPFD = []  # list of solar radiation values
        self.relevant_locations = []
        self.lai_df = pd.read_csv('data/lai.csv')
        self.lai_df = self.lai_df.set_index(['Location'])

    def preprocess(self):  # method for pre-processing data, storing in .csv files
        for filename in os.listdir('Penman_Monteith/sap_flux_um-sec'):
            location = filename.split('_um')[0]
            if location in list(self.lai_df.index.values):
                lai = float(self.lai_df.loc[location, :])
                # Load desired dataframes
                sap_flux_df = pd.read_csv('Penman_Monteith/sap_flux_um-sec/'+location+'_um-sec.csv')
                pm_df = pd.read_csv('Penman_Monteith/pm_prediction/'+location+'_pm_transpiration.csv')
                env_df = pd.read_csv('data/leaf/'+location+'_env_data.csv')
                # Create desired series
                pm_series = pd.Series(pm_df['Reference ET'])
                sf_series = pd.Series(sap_flux_df['Average Sap Flux']*lai)
                ta_series = pd.Series(env_df['ta'])
                rh_series = pd.Series(env_df['rh'])
                vpd_series = pd.Series(env_df['vpd'])
                ppfd_series = pd.Series(env_df['ppfd_in'])
                # if len(pm_series) < 4000:
                #     plt.plot(sap_flux_df['TIMESTAMP'], sap_flux_df['Average Sap Flux'], label='Measurement')
                #     plt.plot(sap_flux_df['TIMESTAMP'], pm_df['Reference ET'], label='Prediction')
                #     plt.legend()
                #     plt.show()

                for i, sf in enumerate(sf_series):
                    if not isnan(sf):  # check that the sap flux value is a number
                        pm = pm_series[i]
                        ta = ta_series[i]
                        rh = rh_series[i]
                        vpd = vpd_series[i]
                        ppfd = ppfd_series[i]
                        if not isnan(pm):  # check that PM is a number
                            # if sf > 0.1:  # causes R^2 to increase by selecting data subset
                            self.PM.append(pm)
                            self.SF.append(sf)
                            if location not in self.relevant_locations:
                                self.relevant_locations.append(location)
                        env_nancheck = [isnan(env) for env in [pm, ta, rh, vpd, ppfd]]
                        if not any(env_nancheck):
                            self.TA.append(ta)
                            self.RH.append(rh)
                            self.VPD.append(vpd)
                            self.PPFD.append(ppfd)
                            self.PM_env.append(pm)
                            self.SF_env.append(sf)
            print(f'{location} done')

        # Save .csv file with PM and SF
        sf_pm_df = pd.DataFrame({'PM': self.PM, 'SF': self.SF})
        sf_pm_df.to_csv('Penman_Monteith/sf_pm_data.csv')

        # Save .csv file with PM_env, SF_env, TA, RH, VPD, PPFD
        env_df = pd.DataFrame({'PM_env': self.PM_env, 'SF_env': self.SF_env, 'TA': self.TA, 'RH': self.RH,
                               'VPD': self.VPD, 'PPFD': self.PPFD})
        env_df.to_csv('Penman_Monteith/env_data.csv')

        print(f'Number of data points is {len(self.PM)}.')
        print(f'Relevant locations were {self.relevant_locations}.')

    def correlation(self):  # method for creating linear regression models
        # Linear regression
        # Sap flux and Penman Monteith comparison
        sf_pm_data = pd.read_csv('Penman_Monteith/sf_pm_data.csv')
        PM = sf_pm_data['PM']
        SF = sf_pm_data['SF']
        lin_reg = sm.OLS(PM, SF).fit()
        print(lin_reg.summary())
        sf_pm_r2 = lin_reg.rsquared
        self.sf_pm_coeff = float(lin_reg.params)
        print(f'R2 for sap flux measurements with the Penman Monteith prediction is {sf_pm_r2} '
              f'and the parameter coefficient is {self.sf_pm_coeff}.')

        # Penman Monteith and environmental variable comparison
        env_data = pd.read_csv('Penman_Monteith/env_data.csv')
        TA = env_data['TA']
        RH = env_data['RH']
        VPD = env_data['VPD']
        PPFD = env_data['PPFD']
        PM_env = env_data['PM_env']
        SF_env = env_data['SF_env']
        env_variables = {'Air temperature': TA, 'Relative humidity': RH, 'Vapor pressure deficit': VPD,
                         'Solar radiation': PPFD}
        for name, var in env_variables.items():
            lin_reg = sm.OLS(PM_env, var).fit()
            r2 = round(lin_reg.rsquared, 5)
            coeff = round(float(lin_reg.params), 5)  # might need to add a constant for some of these and change this
            print(f'R2 with the Penman Monteith prediction for {name} is {r2} '
                  f'and the parameter coefficient is {coeff}.')

        # Sap flux and environmental variable comparison
        for name, var in env_variables.items():
            lin_reg = sm.OLS(SF_env, var).fit()
            r2 = round(lin_reg.rsquared, 5)
            coeff = round(float(lin_reg.params), 5)  # might need to add a constant for some of these and change this
            print(f'R2 with the sap flux measurements for {name} is {r2} and the parameter coefficient is {coeff}.')

    def plotting(self):
        # Plotting
        # Sap flux and Penman Monteith comparison
        sf_pm_data = pd.read_csv('Penman_Monteith/sf_pm_data.csv')
        PM = sf_pm_data['PM']
        SF = sf_pm_data['SF']
        x_reg = np.arange(min(SF), max(SF), 0.1)
        y_reg = self.sf_pm_coeff*x_reg  # use the linear regression coefficient to plot
        plt.scatter(SF, PM)
        plt.plot(x_reg, y_reg, 'r-', label='Linear regression fit')
        plt.xlabel('Sap flux measurement, um/s')
        plt.ylabel('Penman Monteith prediction, um/s')
        plt.legend()
        plt.show()

        # # Penman Monteith with environmental variables
        env_data = pd.read_csv('Penman_Monteith/env_data.csv')
        TA = env_data['TA']
        RH = env_data['RH']
        VPD = env_data['VPD']
        PPFD = env_data['PPFD']
        PM_env = env_data['PM_env']
        SF_env = env_data['SF_env']
        fig, ax = plt.subplots(2, 2, figsize=(8, 8))
        ax[0, 0].scatter(PM_env, VPD)
        ax[0, 0].set_xlabel('Penman Monteith prediction, um/s')
        ax[0, 0].set_ylabel('Vapor pressure deficit')
        ax[0, 1].scatter(PM_env, RH)
        ax[0, 1].set_xlabel('Penman Monteith prediction, um/s')
        ax[0, 1].set_ylabel('Relative humidity, %')
        ax[1, 0].scatter(PM_env, TA)
        ax[1, 0].set_xlabel('Penman Monteith prediction, um/s')
        ax[1, 0].set_ylabel('Air temperature, deg C')
        ax[1, 1].scatter(PM_env, PPFD)
        ax[1, 1].set_xlabel('Penman Monteith prediction, um/s')
        ax[1, 1].set_ylabel('Solar radiation, umol/s*m2')
        plt.show()

        # Sap flux with environmental variables
        fig, ax = plt.subplots(2, 2, figsize=(8, 8))
        ax[0, 0].scatter(SF_env, VPD)
        ax[0, 0].set_xlabel('Sap flux measurements, um/s')
        ax[0, 0].set_ylabel('Vapor pressure deficit')
        ax[0, 1].scatter(SF_env, RH)
        ax[0, 1].set_xlabel('Sap flux measurements, um/s')
        ax[0, 1].set_ylabel('Relative humidity, %')
        ax[1, 0].scatter(SF_env, TA)
        ax[1, 0].set_xlabel('Sap flux measurements, um/s')
        ax[1, 0].set_ylabel('Air temperature, deg C')
        ax[1, 1].scatter(SF_env, PPFD)
        ax[1, 1].set_xlabel('Sap flux measurements, um/s')
        ax[1, 1].set_ylabel('Solar radiation, umol/s*m^2')
        plt.show()


if __name__ == "__main__":
    project = PMsfCompare()
    project.preprocess()
    project.correlation()
    project.plotting()
