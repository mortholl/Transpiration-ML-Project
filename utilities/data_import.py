# Create dictionaries to look up data by time stamp
# Generates the following dictionaries:
# Input features: solar, temp, humidity, vpd, rainfall, soil, gwl_1718, gwl_1920, gwl
# Targets: transpiration_1718, transpiration_1920, transpiration


import pandas as pd
import csv
import statistics


class GatumData:
    def __init__(self):
        # Weather data
        def sanitize_weather(row):
            if all([row[0], row[1], row[2], row[3], row[4]]):
                solar_data = {str(row[0]): float(row[1])}
                temp_data = {str(row[0]): float(row[2])}
                humidity_data = {str(row[0]): float(row[3])}
                vpd_data = {str(row[0]): float(row[4])}
                return solar_data, temp_data, humidity_data, vpd_data
            else:
                return False
        solar = {}
        temp = {}
        humidity = {}
        vpd = {}
        with open('data/weather_30min.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                data = sanitize_weather(row)
                if data:
                    solar.update(data[0])
                    temp.update(data[1])
                    humidity.update(data[2])
                    vpd.update(data[3])
        self.solar = solar
        self.temp = temp
        self.humidity = humidity
        self.vpd = vpd

        # Rainfall data
        rainfall = {}
        with open('data/daily_rain.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                if row:
                    rainfall.update({str(row[0]): float(row[1])})  # daily rainfall in mm
        self.rainfall = rainfall

        # Soil data: global averages between all depths, all locations
        soil_xls = pd.ExcelFile('data/Plantation soil moisture.xls')
        soil_data = pd.read_excel(soil_xls, 'Averages')
        soil_keys = soil_data['Date Time'].to_list()
        soil_values = soil_data['Global'].to_list()
        soil_dict = {}
        for s, date in enumerate(soil_keys):
            soil_dict.update({str(date): float(soil_values[s])})
        self.soil = soil_dict

        # Groundwater level data
        # Use 3663 for 17-18
        gwl_xls = pd.ExcelFile('data/groundwater level.xls')
        gwl = pd.read_excel(gwl_xls, '3663')
        gwl_keys = gwl['datetime'].to_list()
        gwl_values = gwl['waterlevel_corr'].to_list()
        gwl_1718 = {}
        for h, date in enumerate(gwl_keys):
            gwl_1718.update({str(date): float(gwl_values[h])})
        self.gwl_1718 = gwl_1718

        # Use 3666 for 19-20
        gwl_xls = pd.ExcelFile('data/groundwater level.xls')
        gwl = pd.read_excel(gwl_xls, '3666')
        gwl_keys = gwl['datetime'].to_list()
        gwl_values = gwl['waterlevel_corr'].to_list()
        gwl_1920 = {}
        for h, date in enumerate(gwl_keys):
            gwl_1920.update({str(date): float(gwl_values[h])})
        self.gwl_1920 = gwl_1920

        gwl_full = {**gwl_1718, **gwl_1920}
        self.gwl = gwl_full

        # Transpiration data (based on sap flux measurements)
        transpiration_1718 = {}
        transpiration_1920 = {}
        transpiration = {}
        with open('data/avg_transp.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                if row[1]:
                    transpiration_1718.update({str(row[0]): float(row[1])*86400})   # *86,400 to convert to mm/d
                if row[2]:
                    transpiration_1920.update({str(row[0]): float(row[2])*86400})
        transpiration.update(transpiration_1718)
        transpiration.update(transpiration_1920)
        self.transpiration = transpiration
        self.transpiration_1718 = transpiration_1718
        self.transpiration_1920 = transpiration_1920


class AlexWilkieData:   # Daily values --> need to average hourly values to daily
    def __init__(self):
        # Import Alex Wilkie data
        gwl = {}
        with open('data/alex_wilkie_groundwater.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            previous_date = ''
            values = []
            for row in reader:
                if row:
                    date = row[0].split(' ')[0].replace("'", "")
                    if date == previous_date or previous_date == '':
                        values.append(float(row[1]))
                    else:
                        avg = statistics.mean(values)
                        gwl.update({date: avg})   # average groundwater table depth, m
                        values = []
                    previous_date = date
        self.gwl = gwl

        soil_moisture = {}
        with open('data/alex_wilkie_soil_moisture.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            previous_date = ''
            values = []
            for row in reader:
                date = row[0].split(' ')[0].replace("'", "")
                if date == previous_date or previous_date == '':
                    values.append(float(row[1]))
                else:
                    avg = statistics.mean(values)
                    soil_moisture.update({date: avg})  # average mean volumetric soil moisture, m3/m3
                    values = []
                previous_date = date
        self.soil_moisture = soil_moisture

        transpiration = {}
        with open('data/alex_wilkie_transpiration.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                if row:
                    transpiration.update({str(row[0]): float(row[1])})  # average daily transpiration, mm/day
        self.transpiration = transpiration


class NationalDriveData:  # Daily values --> need to average hourly values to daily
    def __init__(self):
        # Import National Drive data
        gwl = {}
        with open('data/national_drive_groundwater.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            previous_date = ''
            values = []
            for row in reader:
                if row:
                    date = row[0].split(' ')[0].replace("'", "")     # split date and time by space, remove the quotes
                    if date == previous_date or previous_date == '':
                        values.append(float(row[1]))
                    else:
                        avg = statistics.mean(values)
                        gwl.update({date: avg})   # average groundwater table depth, m
                        values = []
                    previous_date = date
        self.gwl = gwl

        transpiration = {}
        with open('data/national_drive_transpiration.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                if row:
                    transpiration.update({row[0].replace("'", ""): float(row[1])})  # average daily transpiration, mm/day
        self.transpiration = transpiration

        air_temp = {}
        rainfall = {}
        humidity = {}
        wind = {}
        solar = {}
        vpd = {}
        with open('data/national_drive_weather.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            previous_date = ''
            air_temp_values = []
            rainfall_values = []
            humidity_values = []
            wind_values = []
            solar_values = []
            vpd_values = []
            for row in reader:
                sanitize = True if (row[0], row[1], row[2], row[3], row[5], row[6]) else False
                if sanitize:
                    date = row[0].split(' ')[0].replace("'", "")
                    if date == previous_date or previous_date == '':
                        air_temp_values.append(float(row[1]))
                        rainfall_values.append(float(row[2]))
                        humidity_values.append(float(row[3]))
                        wind_values.append(float(row[4]))
                        solar_values.append(float(row[5]))
                        vpd_values.append(float(row[6]))
                    else:
                        air_temp_avg = statistics.mean(air_temp_values)
                        rainfall_total = sum(rainfall_values)
                        humidity_avg = statistics.mean(humidity_values)
                        wind_avg = statistics.mean(wind_values)
                        solar_avg = statistics.mean(solar_values)
                        vpd_avg = statistics.mean(vpd_values)
                        air_temp.update({date: air_temp_avg})
                        rainfall.update({date: rainfall_total})
                        humidity.update({date: humidity_avg})
                        wind.update({date: wind_avg})
                        solar.update({date: solar_avg})
                        vpd.update({date: vpd_avg})
                        air_temp_values = []
                        rainfall_values = []
                        humidity_values = []
                        wind_values = []
                        solar_values = []
                        vpd_values = []
                    previous_date = date
        self.air_temp = air_temp      # Average daily air temperature, deg C
        self.rainfall = rainfall      # Cumulative daily rainfall, mm
        self.humidity = humidity      # Average daily relative humidity, %
        self.wind = wind              # Average daily wind speed, m/s
        self.solar = solar            # Average daily solar radiation, W/m2
        self.vpd = vpd                # Average daily vapor pressure deficit, kPa


if __name__ == "__main__":
    test = NationalDriveData()
