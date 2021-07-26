from Penman_Monteith import equations
from utilities.data_import import GatumData

Data = GatumData()
# Radiation [W/m2], Temperature [K], Specific Humidity [kg/kg]

pm_humidity = Data.humidity  # %
pm_temp = Data.temp  # deg C
pm_solar = Data.solar  # W/m2


# Loop over times (keys) to sanitize missing data, convert humidity and temp, return PM transpiration

pm_transpiration = {}
for time, value in pm_humidity.items():
    if value and pm_temp[time] and pm_solar[time]:
        phi = pm_solar[time]
        ta = pm_temp[time] + 273
        qa = equations.qaRh(value, ta)
        transpiration = equations.evfPen(phi, qa, ta)
        pm_transpiration.update({time: transpiration})

