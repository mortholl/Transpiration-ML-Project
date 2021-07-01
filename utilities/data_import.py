# Create dictionaries to look up data by time stamp
# Generates the following dictionaries:
# Input features: solar, temp, humidity, vpd, rainfall, soil, gwl_1718, gwl_1920, gwl
# Targets: transpiration_1718, transpiration_1920, transpiration


import pandas as pd
import csv


class DataDict:
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
        transpiration_1718 = []
        transpiration_1920 = []
        with open('data/avg_transp.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                if row[1]:
                    transpiration_1718.append([str(row[0]), float(row[1])])
                if row[2]:
                    transpiration_1920.append([str(row[0]), float(row[2])])
        transpiration = transpiration_1718 + transpiration_1920
        self.transpiration = transpiration
        self.transpiration_1718 = transpiration_1718
        self.transpiration_1920 = transpiration_1920


if __name__ == "__main__":
    test = DataDict()
