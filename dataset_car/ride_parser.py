import os
import pandas as pd
import matplotlib.pyplot as plt

import io
import json
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import os

import numpy as np
import math

import pyproj
import matplotlib.dates as mdates

import re

import folium

import requests


# Fri Nov 03 13:37:58 GMT-03:00 2023
old_app_date_format = "%a %b %d %H:%M:%S %Z%z %Y"

# "yyyy-MM-dd HH:mm:ss.SSS"
new_app_date_format = "%Y-%m-%d %H:%M:%S.%f"

# def std_param_name(name):
# 	name = name.strip()
# 	name = re.sub(r"\s+", "_", name)
# 	name = re.sub(r"[^$/\w]", "", name)
# 	return name.lower()

# def cartesian_product(*arrays):
#     la = len(arrays)
#     dtype = np.result_type(*arrays)
#     arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
#     for i, a in enumerate(np.ix_(*arrays)):
#         arr[...,i] = a
#     return arr.reshape(-1, la)

# definir raio de relevancia:
# https://www.prefeitura.sp.gov.br/cidade/secretarias/subprefeituras/subprefeituras/dados_demograficos/index.php?p=12758

def get_dist_from_coords(coords_1, coords_2):
	lat_1 = coords_1[0]
	long_1 = coords_1[1]

	lat_2 = coords_2[0]
	long_2 = coords_2[1]

	delta_lat = (lat_2 - lat_1) * math.pi / 180
	delta_long = (long_1 - long_2) * math.pi / 180

	# https://stackoverflow.com/questions/639695/how-to-convert-latitude-or-longitude-to-meters
	R = 6378.137 # radius of earth in KM

	a_1 = math.sin(delta_lat/2) ** 2
	a_2 = math.cos(lat_1 * math.pi / 180) * math.cos(lat_2 * math.pi / 180) * math.sin(delta_long/2) ** 2
	a = a_1 + a_2

	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
	delta_distance = (R * c) * 1000 # meters

	return delta_distance

class CrimeAnalyser():

	def __init__(self):
		
		# https://www.kaggle.com/datasets/danlessa/geospatial-sao-paulo-crime-database/
		# https://www.kaggle.com/code/anagagodasilva/s-o-paulo-crime-maps-with-plotly/notebook
		sp_crimes_path = os.path.join("CrimeData", "crimes_por_bairro_sao_paulo.csv")

		sp_crimes_df = pd.read_csv(sp_crimes_path)
		car_crimes_df = sp_crimes_df[sp_crimes_df["descricao"].str.contains("carro", na=False)].copy()

		# https://brasilemsintese.ibge.gov.br/territorio/dados-geograficos.html
		# https://www.teleco.com.br/tutoriais/tutorialsmsloc2/pagina_5.asp#:~:text=Cada%20grau%20de%20uma%20latitude,1%C2%B0%20(um%20grau).
		# give 0.5km margin for furthest chunk boundaries
		self.north = car_crimes_df["latitude"].max() + (0.5 / 111.11) # brasil: 5.27194444444  =	+05o 16"19"
		self.south = car_crimes_df["latitude"].min() - (0.5 / 111.11) # brasil: -33.7519444444 =	-33o 45"07"
		self.west = car_crimes_df["longitude"].max() + (0.5 / 111.11) # brasil: -73.9905555556 = 	-73o 59"26"
		self.east = car_crimes_df["longitude"].min() - (0.5 / 111.11) # brasil: -34.7927777778 =	-34o 47"34"

		# https://brasilescola.uol.com.br/brasil/pontos-extremos-do-brasil.htm
		# biggest north-south dist: 4378.4 km
		# biggest east-west dist: 4326.6 km
		self.segmentation_rate = 4400

		self.lat_diff = self.north - self.south
		self.long_diff = self.east - self.west

		
		car_crimes_df["chunk_i"] = car_crimes_df["latitude"].apply(lambda x : self.convert_coord_to_chunk(x, self.south, self.lat_diff))
		car_crimes_df["chunk_j"] = car_crimes_df["longitude"].apply(lambda x : self.convert_coord_to_chunk(x, self.west, self.long_diff))

		car_crimes_df["crime_count_in_chunk"] = car_crimes_df[["chunk_i", "chunk_j"]].groupby(by=["chunk_i", "chunk_j"]).transform("size")

		self.min_danger_boundary = 5 # crimes per chunk
		self.max_danger_boundary = 9 # crimes per chunk
		
		self.car_crimes_df = car_crimes_df

	def get_chunk_df(self, lat, long):
		chunk_i = self.convert_coord_to_chunk(lat, self.south, self.lat_diff)
		chunk_j = self.convert_coord_to_chunk(long, self.west, self.long_diff)

		# TODO: check if coord dist of chunks corresponds to needed diameter for analysis

		filter_i = (self.car_crimes_df["chunk_i"] - chunk_i).abs() <= 1
		filter_j = (self.car_crimes_df["chunk_j"] - chunk_j).abs() <= 1

		return self.car_crimes_df[filter_i & filter_j]

	def get_chunk_danger(self, lat, long):
		chunk_df = self.get_chunk_df(lat, long)
		danger_list = []
		for _, crime_count_series in chunk_df[["crime_count_in_chunk"]].iterrows():
			crime_count = int(crime_count_series.iloc[0])
			if crime_count <= self.min_danger_boundary:
				danger_list.append(1)
			elif crime_count >= self.max_danger_boundary:
				danger_list.append(3)
			else:
				danger_list.append(2)

		if len(danger_list) == 0:
			return 1
		return math.floor(np.average(danger_list))

	def convert_coord_to_chunk(self, coord_1, coord_2, max_coord_diff):
		coord_diff = coord_1 - coord_2
		chunk = (self.segmentation_rate * coord_diff) // max_coord_diff
		return chunk


class RealRideParser():

	car_crimes_df = None
	lambda_url = "https://pntdpvkdsc.execute-api.us-east-1.amazonaws.com/default/app_data"

	def __init__(self, should_get_data_from_database=False, **params):

		RealRideParser.crime_analyser = CrimeAnalyser()

		if should_get_data_from_database:
			self.user_id = params["user_id"]
			self.date_beg = params["date_beg"]
			self.date_end = params["date_end"]
		else:
			self.root_dir = params["root_dir"]

		self.should_get_data_from_database = should_get_data_from_database

		# TODO: remover outliers antes de tudo

		self.obd_data = self.create_obd_df()
		self.gps_df = self.create_gps_df()
		self.accelerometer_df = self.create_accelerometer_df()

		if not should_get_data_from_database:
			self.orientation_df = self.create_orientation_df()
			self.bearing_df = self.create_bearing_df()

	def generate_pdf_metrics(self):
		map = self.create_route_map()
		map, danger_list = self.calculate_crime_stats(map)
		risk_table_graph = self.generate_risk_table(danger_list)
		# risk_table_graph.savefig("bla.png")
		# self.calculate_acc_stats_near_stop()
		map, sudden_acc_table = self.generate_sudden_acc(map)
		sudden_acc_table.show()
		# print("Porcentagem de acelerações acima do normal: ", sudden_acc_percentage)
		map, excess_rpm_table = self.generate_rpm_graph(map)
		excess_rpm_table.show()

		return map

	def generate_rpm_graph(self, map):
		gps_df = self.gps_df
		rpm_df = self.obd_data["ENGINE_RPM"]

		data = []
		for _, rpm in rpm_df.iterrows():
			timestamp = rpm["timestamp"]
			rpm_val = rpm["ENGINE_RPM"]
			gps_id = min(len(gps_df) - 1, gps_df["timestamp"].searchsorted(timestamp))

			lat = gps_df.iloc[gps_id]["lat"]
			long = gps_df.iloc[gps_id]["long"]

			data.append({
				"timestamp": timestamp,
				"lat": lat,
				"long": long,
				"rpm": rpm_val
			})

		all_rpm_df = pd.DataFrame(data)

		dangerous_rpm_list = []

		# excess_rpm_df = all_rpm_df[all_rpm_df["resultant"] >= 5]
		for _, rpm in all_rpm_df.iterrows():

			lat = rpm["lat"]
			long = rpm["long"]
			
			rpm_resultant = float(rpm["rpm"])
			if rpm_resultant <= 2000:
				dangerous_rpm_list.append(1)
			elif rpm_resultant >= 4000:
				dangerous_rpm_list.append(3)
			else:
				dangerous_rpm_list.append(2)

			danger_to_color = {
				1: "green",
				2: "orange",
				3: "red",
			}			
			color = danger_to_color[dangerous_rpm_list[-1]]

			folium.Marker([lat, long], icon=folium.Icon(icon="gear", prefix="fa", color=color)).add_to(map)

		# excess_rpm_percentage = len(excess_rpm_df) / len(all_rpm_df)
		# excess_rpm_percentage = "%.2f" % (excess_rpm_percentage * 100)

		level_to_name = {
			1: "RPM normal",
			2: "RPM médio",
			3: "RPM muito alto"
		}
		danger_percentage_df = pd.DataFrame({"danger_level": dangerous_rpm_list})
		danger_percentage_df["danger_name"] = danger_percentage_df["danger_level"].map(level_to_name)

		count_df = danger_percentage_df["danger_name"].value_counts().reset_index()
		count_df.columns = ["danger_name", "count"]


		default_rows = pd.DataFrame({"danger_name": ["RPM normal", "RPM médio", "RPM muito alto"],
									"count": [0, 0, 0]})

		merged_df = pd.merge(default_rows, count_df, on="danger_name", how="left").fillna(0)
		merged_df["count"] = merged_df[["count_x", "count_y"]].max(axis=1)

		percentage_series = merged_df['count'] / merged_df["count"].sum()
		merged_df["percentage"] = percentage_series.apply(lambda x : "%.2f%%" % (x * 100))

		fig, ax = plt.subplots(figsize=(5, 1))

		ax.xaxis.set_visible(False)
		ax.yaxis.set_visible(False)
		ax.set_frame_on(False)

		cell_text = []
		for i in range(3):
			cell_text.append([merged_df.iloc[i]["percentage"]])

		tab = plt.table(cellText=cell_text, rowLabels=merged_df["danger_name"], colLabels=["Fração do tempo com cada tipo de RPM"], loc="center", colWidths=[1, 1.1], cellLoc="center")
		tab.auto_set_font_size(False)
		tab.set_fontsize(10)
		tab.scale(1, 2)
		# ax.set_title("Contagem de Níveis de Perigo")

		return map, plt.gcf()

	def generate_sudden_acc(self, map):
		gps_df = self.gps_df
		resultant_acc_df = self.accelerometer_df[["timestamp", "remaining_acc_resultant"]]

		data = []
		for _, acc in resultant_acc_df.iterrows():
			timestamp = acc["timestamp"]
			resultant_acc = acc["remaining_acc_resultant"]
			gps_id = min(len(gps_df) - 1, gps_df["timestamp"].searchsorted(timestamp))

			lat = gps_df.iloc[gps_id]["lat"]
			long = gps_df.iloc[gps_id]["long"]

			data.append({
				"timestamp": timestamp,
				"lat": lat,
				"long": long,
				"resultant": resultant_acc
			})

		all_acc_df = pd.DataFrame(data)

		dangerous_acc_list = []

		for _, acc in all_acc_df.iterrows():

			lat = acc["lat"]
			long = acc["long"]
			
			acc_resultant = float(acc["resultant"])
			if acc_resultant <= 5:
				dangerous_acc_list.append(1)
			elif acc_resultant >= 10:
				dangerous_acc_list.append(3)
			else:
				dangerous_acc_list.append(2)

			danger_to_color = {
				1: "green",
				2: "orange",
				3: "red",
			}			
			color = danger_to_color[dangerous_acc_list[-1]]

			folium.Marker([lat, long], icon=folium.Icon(icon="car", prefix="fa", color=color)).add_to(map)

		# sudden_acc_df = all_acc_df[all_acc_df["resultant"] >= 5]
		# sudden_acc_percentage = len(sudden_acc_df) / len(all_acc_df)
		# sudden_acc_percentage = "%.2f" % (sudden_acc_percentage * 100)

		level_to_name = {
			1: "Aceleração normal",
			2: "Aceleração média",
			3: "Aceleração muito alta"
		}
		danger_percentage_df = pd.DataFrame({"danger_level": dangerous_acc_list})
		danger_percentage_df["danger_name"] = danger_percentage_df["danger_level"].map(level_to_name)

		count_df = danger_percentage_df["danger_name"].value_counts().reset_index()
		count_df.columns = ["danger_name", "count"]


		default_rows = pd.DataFrame({"danger_name": ["Aceleração normal", "Aceleração média", "Aceleração muito alta"],
									"count": [0, 0, 0]})

		merged_df = pd.merge(default_rows, count_df, on="danger_name", how="left").fillna(0)
		merged_df["count"] = merged_df[["count_x", "count_y"]].max(axis=1)

		percentage_series = merged_df['count'] / merged_df["count"].sum()
		merged_df["percentage"] = percentage_series.apply(lambda x : "%.2f%%" % (x * 100))

		fig, ax = plt.subplots(figsize=(5, 1))

		ax.xaxis.set_visible(False)
		ax.yaxis.set_visible(False)
		ax.set_frame_on(False)

		cell_text = []
		for i in range(3):
			cell_text.append([merged_df.iloc[i]["percentage"]])

		tab = plt.table(cellText=cell_text, rowLabels=merged_df["danger_name"], colLabels=["Fração do tempo com cada tipo de aceleração"], loc="center", colWidths=[1, 1.1], cellLoc="center")
		tab.auto_set_font_size(False)
		tab.set_fontsize(10)
		tab.scale(1, 2)
		# ax.set_title("Contagem de Níveis de Perigo")

		return map, plt.gcf()


	def generate_risk_table(self, danger_list):
		level_to_name = {
			1: "Baixo Risco",
			2: "Médio Risco",
			3: "Alto Risco"
		}
		danger_percentage_df = pd.DataFrame({"danger_level": danger_list})
		danger_percentage_df["danger_name"] = danger_percentage_df["danger_level"].map(level_to_name)

		count_df = danger_percentage_df["danger_name"].value_counts().reset_index()
		count_df.columns = ["danger_name", "count"]


		default_rows = pd.DataFrame({"danger_name": ["Baixo Risco", "Médio Risco", "Alto Risco"],
									"count": [0, 0, 0]})

		merged_df = pd.merge(default_rows, count_df, on="danger_name", how="left").fillna(0)
		merged_df["count"] = merged_df[["count_x", "count_y"]].max(axis=1)

		percentage_series = merged_df['count'] / merged_df["count"].sum()
		merged_df["percentage"] = percentage_series.apply(lambda x : "%.2f%%" % (x * 100))

		fig, ax = plt.subplots(figsize=(5, 1))

		ax.xaxis.set_visible(False)
		ax.yaxis.set_visible(False)
		ax.set_frame_on(False)

		cell_text = []
		for i in range(3):
			cell_text.append([merged_df.iloc[i]["percentage"]])

		tab = plt.table(cellText=cell_text, rowLabels=merged_df["danger_name"], colLabels=["Fração do tempo passada lá"], loc="center", colWidths=[1, 1.1], cellLoc="center")
		tab.auto_set_font_size(False)
		tab.set_fontsize(10)
		tab.scale(1, 2)
		# ax.set_title("Contagem de Níveis de Perigo")

		return plt.gcf()


	def calculate_crime_stats(self, map):
		
		path_latlongs = []
		for path_lat_long in self.gps_df[["lat", "long"]].iterrows():
			path_lat_long = tuple(path_lat_long[1])

			danger_level = RealRideParser.crime_analyser.get_chunk_danger(*path_lat_long)
			path_latlongs.append((path_lat_long, danger_level))


			# crimes_in_chunk_df = RealRideParser.crime_analyser.get_chunk_df(*path_lat_long)

			# for _, reported_crime_lat_long in crimes_in_chunk_df[["latitude", "longitude"]].iterrows():
			# 	reported_crime_lat_long = tuple(reported_crime_lat_long)
			# 	dist_from_target = get_dist_from_coords(path_lat_long, reported_crime_lat_long)

			# 	# TODO: calcular proporcao de crimes num raio de 100 metros em comparação com um raio de 300m
			# 	if dist_from_target <= 500:
			# 		# print(reported_crime_lat_long, dist_from_target)
			# 		path_latlongs.append(path_lat_long)

		danger_list = []
		for latlong, danger in path_latlongs:
			danger_list.append(danger)

			if danger <= 1:
				continue

			danger_to_color = {
				1: "green",
				2: "orange",
				3: "red",
			}			
			color = danger_to_color[danger]
			
			folium.Marker(latlong, icon=folium.Icon(icon="circle-exclamation", prefix="fa", color=color)).add_to(map)

		return map, danger_list


	def create_route_map(self):
		latitudes = self.gps_df["lat"]
		longitudes = self.gps_df["long"]

		qtty_data = len(latitudes)

		lat_longs = [(latitudes.iloc[i], longitudes.iloc[i]) for i in range(qtty_data)]

		start_lat_long = lat_longs[0]
		end_lat_long = lat_longs[-1]

		mean_lat = latitudes.mean()
		mean_long = longitudes.mean()

	
		map = folium.Map(location=[mean_lat, mean_long], zoom_start=14, control_scale=True)

		folium.Marker(start_lat_long, popup="start", icon=folium.Icon(icon="flag_checkered", prefix="fa", color="blue")).add_to(map)
		folium.Marker(end_lat_long, popup="end", icon=folium.Icon(icon="x", prefix="fa", color="red")).add_to(map)

		for i in range(qtty_data - 1):
			location_1 = lat_longs[i]
			location_2 = lat_longs[i + 1]

			folium.PolyLine([location_1, location_2],
							color="red",
							weight=5,
							opacity=0.4).add_to(map)

		return map

	def calculate_acc_stats_near_stop(self):
		vels_from_obd_df = self.obd_data["SPEED"]
		acc_from_android_df = self.accelerometer_df

		stopped_speeds = vels_from_obd_df[vels_from_obd_df["SPEED"] <= 0.1]
		stopped_speeds_indices = stopped_speeds.index
		qtty_of_indices = len(stopped_speeds_indices)

		acceleration_starts = []
		acceleration_endings = [0]
		for i in range(qtty_of_indices - 1):
			row_id_1 = stopped_speeds_indices[i]
			row_id_2 = stopped_speeds_indices[i + 1]

			if row_id_1 != row_id_2 - 1:
				acceleration_starts.append(row_id_1)
				acceleration_endings.append(row_id_2)

		acceleration_starts.append(stopped_speeds_indices[-1])

		acceleration_start_timestamps = vels_from_obd_df.filter(items=acceleration_starts, axis=0)["timestamp"]
		acceleration_ending_timestamps = vels_from_obd_df.filter(items=acceleration_endings, axis=0)["timestamp"]

		####################################

		print(acceleration_starts)
		print(len(acceleration_starts))
		print(len(acceleration_start_timestamps))
		# acc_from_android_df
		# for timestamp in acceleration_start_timestamps:

		re_acceleration_avg_s = []
		last_acc_start_id = 0
		# cut off first from endings and first from starts
		for i in range(len(acceleration_starts) - 2):
			# print(i)

			non_zero_vel_beg_id = acceleration_starts[i]
			non_zero_vel_end_id = acceleration_endings[i + 1]

			start_timestamp = vels_from_obd_df.iloc[non_zero_vel_beg_id]["timestamp"]
			
			# vel_index = 

			acc_list = []
			for j in range(non_zero_vel_beg_id, non_zero_vel_end_id - 1):
				vel_1 = vels_from_obd_df.iloc[j]["SPEED"]
				vel_2 = vels_from_obd_df.iloc[j + 1]["SPEED"]

				if vel_1 > vel_2:
					break

				acc_index = acc_from_android_df["timestamp"].searchsorted(start_timestamp)
				new_acc = acc_from_android_df["acc_resultant"].iloc[acc_index]

				acc_list.append(new_acc)

			last_acc_start_id = non_zero_vel_beg_id #acc_index

			re_acceleration_avg = np.average(acc_list)
			re_acceleration_avg_s.append(re_acceleration_avg)
			
			# print("acc_list", acc_list)
			# print("re_acceleration_avg", re_acceleration_avg)

		# 	acc_from_android_df.iloc[acc_index]["acceleration"]

		# acceleration_start_timestamps, 
		print(np.average(re_acceleration_avg_s))
		print(np.std(re_acceleration_avg_s))
		print(np.mean(re_acceleration_avg_s))
		print(np.median(re_acceleration_avg_s))
		print(np.quantile(re_acceleration_avg_s, 0.25))
		print(np.quantile(re_acceleration_avg_s, 0.75))
		print(min(re_acceleration_avg_s))
		print(max(re_acceleration_avg_s))

	def generate_temp_graphs(self):
		n_params = len(self.temp_params)
		fig, axs = plt.subplots(ncols=1, nrows=n_params)

		for i in range(n_params):
			param = self.temp_params[i]

			df_to_display = self.obd_data[param]
			timestamp = df_to_display["timestamp"]
			param_series = df_to_display[param]

			axs[i].plot(timestamp, param_series, label=param)
			axs[i].legend(loc="right")
			axs[i].set_title(param)

		fig.tight_layout()
		plt.grid(True)

		# plt.show()
		return plt.gcf()

	def generate_pressure_graphs(self):
		n_params = len(self.pressure_params)
		fig, axs = plt.subplots(ncols=1, nrows=n_params)

		for i in range(n_params):
			param = self.pressure_params[i]

			df_to_display = self.obd_data[param]
			timestamp = df_to_display["timestamp"]
			param_series = df_to_display[param]

			axs[i].plot(timestamp, param_series, label=param)
			axs[i].legend(loc="right")
			axs[i].set_title(param)

		fig.tight_layout()
		plt.grid(True)

		# plt.show()
		return plt.gcf()

	def generate_other_graphs(self):
		n_params = len(self.other_params)
		fig, axs = plt.subplots(ncols=1, nrows=n_params)

		for i in range(n_params):
			param = self.other_params[i]

			df_to_display = self.obd_data[param]
			timestamp = df_to_display["timestamp"]
			param_series = df_to_display[param]

			axs[i].plot(timestamp, param_series, label=param)
			axs[i].legend(loc="right")
			axs[i].set_title(param)

		fig.tight_layout()
		plt.grid(True)

		# plt.show()
		return plt.gcf()

	def create_obd_df(self):
		if self.should_get_data_from_database:
			request_body = {
				"method": "get_obd_info",
				"user_token": self.user_id,
				"time_min": self.date_beg,
				"time_max": self.date_end
			}
			
			response = requests.post(RealRideParser.lambda_url, json=request_body)
			response_dict = json.loads(response.text)

			engine_data_list = []
			for data in response_dict["data"]:
				response_list = data[1:]
				corrected_list = ["" if x is None else x for x in response_list]
				engine_data = corrected_list[0] + " " + json.dumps([corrected_list[2], corrected_list[2], corrected_list[3]])
				engine_data_list.append(engine_data)

			if len(engine_data_list) == 0:
				return pd.DataFrame()
			
			engine_data = "\n".join(engine_data_list)
		else:
			engine_data_path = os.path.join(self.root_dir, "DELETEME.txt")
			engine_data = open(engine_data_path, "r").read().strip()

		param_name_to_df = {}
		for data_entry in engine_data.split("\n"):
			if data_entry == "":
				continue

			data_entry = data_entry.replace("NODATA", "0")

			try:
				date = data_entry[:34]
				info_list = json.loads(data_entry[35:])
				data_obj = datetime.strptime(date, old_app_date_format)
			except Exception as e:
				date = data_entry[:23]
				info_list = json.loads(data_entry[24:])
				data_obj = datetime.strptime(date, new_app_date_format)


			param_name = info_list[0]
			param_value = info_list[2]

			timestamp = data_obj.timestamp()

			create_default_df = (lambda : pd.DataFrame(columns=["timestamp", param_name]))
			param_df = param_name_to_df.get(param_name, create_default_df())

			param_df.loc[len(param_df.index)] = [timestamp, param_value]
			param_name_to_df[param_name] = param_df

		speed_series = param_name_to_df["SPEED"]["SPEED"]
		
		def convert_vel_to_float(x):
			vel = x.replace("km/h", "")
			if vel == "":
				return 0
			return float(vel)
		
		def convert_rpm_to_float(x):
			rpm = x.replace("RPM", "")
			if rpm == "":
				return 0
			try:
				return float(rpm)
			except:
				print("invalid rpm: %s" % rpm)
				return 0
		
		param_name_to_df["SPEED"]["SPEED"] = speed_series.apply(convert_vel_to_float)
		param_name_to_df["ENGINE_RPM"]["ENGINE_RPM"] = param_name_to_df["ENGINE_RPM"]["ENGINE_RPM"].apply(convert_rpm_to_float)

		def convert_temp_to_float(x):
			temp = x.replace("C", "")
			if temp == "":
				return 0
			try:
				return float(temp)
			except:
				print("cannot convert temp %s to float" % temp)
				return 0

		def convert_pressure_to_float(x):
			pressure = x.replace("kPa", "")
			if pressure == "":
				return 0
			try:
				return float(pressure)
			except:
				print("cannot convert pressure %s to float" % pressure)
				return 0
		temp_params = []
		pressure_params = []
		other_params = []
		for param_name in param_name_to_df:
			standardized_param_name = param_name.lower()
			param_series = param_name_to_df[param_name][param_name]

			if "temp" in standardized_param_name:
				temp_params.append(param_name)

				param_name_to_df[param_name][param_name] = param_series.apply(convert_temp_to_float)

			elif "pressure" in standardized_param_name:
				pressure_params.append(param_name)

				param_name_to_df[param_name][param_name] = param_series.apply(convert_pressure_to_float)

			else:
				other_params.append(param_name)

				# param_name_to_df[param_name][param_name] = param_series.apply(convert_other_to_float)


		self.temp_params = temp_params
		self.pressure_params = pressure_params
		self.other_params = other_params

		# immutable_data = []
		# mutable_data = [] 
		# for param_name, df in param_name_to_df.items():
		# 	all_values_set = set(df[param_name])
		# 	if len(all_values_set) <= 2:
		# 		immutable_data.append(param_name)
		# 	else:
		# 		mutable_data.append(param_name)

		# print("immutable_data")
		# for data in immutable_data:
		# 	print(data)

		# print("mutable_data")
		# for data in mutable_data:
		# 	print(data)


		return param_name_to_df

	def create_velocity_from_gps_df(self):
		gps_df = self.gps_df #.drop_duplicates(subset=["timestamp"])
		n_samples = len(gps_df)

		data = []
		desired_cols = ["timestamp", "lat", "long"]
		for i in range(n_samples - 1):
			time_1, lat_1, long_1 = tuple(gps_df[desired_cols].iloc[i])
			time_2, lat_2, long_2 = tuple(gps_df[desired_cols].iloc[i + 1])

			delta_t = time_2 - time_1 # seconds
			delta_lat = (lat_2 - lat_1) * math.pi / 180
			delta_long = (long_1 - long_2) * math.pi / 180

			if delta_t == 0:
				continue

			# https://stackoverflow.com/questions/639695/how-to-convert-latitude-or-longitude-to-meters
			R = 6378.137 # radius of earth in KM

			a_1 = math.sin(delta_lat/2) ** 2
			a_2 = math.cos(lat_1 * math.pi / 180) * math.cos(lat_2 * math.pi / 180) * math.sin(delta_long/2) ** 2
			a = a_1 + a_2

			c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
			delta_distance = (R * c) * 1000 # meters

			v_meters_p_second = delta_distance / delta_t
			v_kmh = v_meters_p_second * 3.6
					
			data_line = {
				"timestamp": time_1,
				"SPEED": v_kmh
			}

			data.append(data_line)

		velocity_df = pd.DataFrame(data)

		return velocity_df

	def get_acc_stats(self):
		acc_df = self.accelerometer_df

		return acc_df["acc_resultant"].describe() #tats
	
	def parse_data_line(self, line):
		try:
			actual_data = json.loads(line[35:])
			dot_convert_foo = lambda x: float(x.replace(",", "."))
			actual_data = list(map(dot_convert_foo, actual_data))
			original_time_string = line[:34]
		except:
			actual_data = json.loads(line[24:])
			dot_convert_foo = lambda x: float(x.replace(",", "."))
			actual_data = list(map(dot_convert_foo, actual_data))
			original_time_string = line[:23]

		try:
			data_date = datetime.strptime(original_time_string, old_app_date_format)
		except:
			data_date = datetime.strptime(original_time_string, new_app_date_format)

		timestamp = data_date.timestamp()

		parsed_data_list = [timestamp] + actual_data
		return tuple(parsed_data_list)


	def create_orientation_df(self):
		if self.should_get_data_from_database:
			pass
		else:
			orientation_file_path = os.path.join(self.root_dir, "DELETEME_ORIENTATION.txt")
			orientation_file = open(orientation_file_path, "r")

		data = []
		for line in orientation_file.readlines():

			timestamp, azimuth, pitch, roll = self.parse_data_line(line)

			data_line = {
				"timestamp": timestamp,
				"azimuth": float(azimuth),
				"pitch": float(pitch),
				"roll": float(roll)
			}

			data.append(data_line)

		return pd.DataFrame(data)


	def create_bearing_df(self):
		if self.should_get_data_from_database:
			pass
		else:
			bearing_file_path = os.path.join(self.root_dir, "DELETEME_BEARING.txt")
			bearing_file = open(bearing_file_path, "r")

		data = []
		for line in bearing_file.readlines():

			timestamp, angle = self.parse_data_line(line)

			data_line = {
				"timestamp": timestamp,
				"angle": float(angle)
			}

			data.append(data_line)

		return pd.DataFrame(data)


	def create_gps_df(self):
		if self.should_get_data_from_database:
			request_body = {
				"method": "get_location",
				"user_token": self.user_id,
				"time_min": self.date_beg,
				"time_max": self.date_end
			}
			
			response = requests.post(RealRideParser.lambda_url, json=request_body)
			response_dict = json.loads(response.text)

			gps_data_list = []
			for data in response_dict["data"]:
				response_list = data[1:]
				corrected_list = ["" if x is None else x for x in response_list]
				gps_data = corrected_list[0] + " " + json.dumps(corrected_list[2:])
				gps_data_list.append(gps_data)

			if len(gps_data_list) == 0:
				return pd.DataFrame()
			
			gps_data = "\n".join(gps_data_list)
		else:
			gps_file_path = os.path.join(self.root_dir, "DELETEME_GPS.txt")
			gps_data= open(gps_file_path, "r").read().strip()

		data = []

		for line in gps_data.split("\n"):
			if line == "":
				continue

			timestamp, lat, long = self.parse_data_line(line)

			data_line = {
				"timestamp": timestamp,
				"lat": float(lat),
				"long": float(long)
			}

			data.append(data_line)

		return pd.DataFrame(data)

	def create_accelerometer_df(self):		
		if self.should_get_data_from_database:
			request_body = {
				"method": "get_acceleration",
				"user_token": self.user_id,
				"time_min": self.date_beg,
				"time_max": self.date_end
			}
			
			response = requests.post(RealRideParser.lambda_url, json=request_body)
			response_dict = json.loads(response.text)

			acc_data_list = []
			for data in response_dict["data"]:
				response_list = data[1:]
				corrected_list = ["" if x is None else x for x in response_list]
				acc_data = corrected_list[0] + " " + json.dumps(corrected_list[2:])
				acc_data_list.append(acc_data)

			if len(acc_data_list) == 0:
				return pd.DataFrame()
			
			acc_data = "\n".join(acc_data_list)
		else:
			accelerometer_file_path = os.path.join(self.root_dir, "DELETEME_ACCELERATION.txt")
			acc_data= open(accelerometer_file_path, "r").read().strip()

		data = []

		for line in acc_data.split("\n"):
			if line == "":
				continue

			parsed_data = self.parse_data_line(line)

			timestamp = parsed_data[0]
			acc_s = np.array(parsed_data[1:4])
			gravity_vec = np.array(parsed_data[4:7])

			acc_z_car = np.dot(acc_s, gravity_vec) * (gravity_vec / np.linalg.norm(gravity_vec))
			remaining_acc = acc_s - acc_z_car

			data_line = {
				"timestamp": timestamp,
				"acc_x": acc_s[0],
				"acc_y": acc_s[1],
				"acc_z": acc_s[2],
				"filtered_acc_x": acc_s[0],
				"filtered_acc_y": acc_s[1],
				"filtered_acc_z": acc_s[2],
				"grav_x": gravity_vec[0],
				"grav_y": gravity_vec[1],
				"grav_z": gravity_vec[2],
				"acc_z_car_x": acc_z_car[0],
				"acc_z_car_y": acc_z_car[1],
				"acc_z_car_z": acc_z_car[2],
				"remaining_acc_x": remaining_acc[0],
				"remaining_acc_y": remaining_acc[1],
				"remaining_acc_z": remaining_acc[2]
			}

			data.append(data_line)

		acc_df = pd.DataFrame(data)

		acc_df["acc_resultant"] = np.sqrt(acc_df["acc_x"] ** 2 + acc_df["acc_y"] ** 2 + acc_df["acc_z"] ** 2)
		acc_df["remaining_acc_resultant"] = np.sqrt(acc_df["remaining_acc_x"] ** 2 + acc_df["remaining_acc_y"] ** 2 + acc_df["remaining_acc_z"] ** 2)

		return acc_df

	def generate_3_axis_acc_graph(self, frame_granularity, delta_time_size, img_id):
		fig, axs = plt.subplots(ncols=1, nrows=4)

		accelerometer_df = self.accelerometer_df

		min_timestamp = accelerometer_df["timestamp"].iloc[0] #d.timestamp()
		# max_timestamp = accelerometer_df["timestamp"].iloc[-1]

		lower_limit_timestamp = min_timestamp + frame_granularity * img_id
		lower_limit = lower_limit_timestamp
		upper_limit = lower_limit_timestamp + delta_time_size
		# print("low up", lower_limit, upper_limit)
		# print("min max", min_timestamp, max_timestamp)

		# import pdb; pdb.set_trace()

		timestamps_series = accelerometer_df["timestamp"] #.apply(lambda x : x.timestamp())
		time_filter = (timestamps_series >= lower_limit) & (timestamps_series < upper_limit)
		filtered_accelerations_df = accelerometer_df[time_filter]
		# print("filtered_accelerations_df", filtered_accelerations_df)
		# print("accelerometer_df", accelerometer_df)

		acc_x = filtered_accelerations_df["acc_x"]
		acc_y = filtered_accelerations_df["acc_y"]
		acc_z = filtered_accelerations_df["acc_z"]
		acc_resultant = filtered_accelerations_df["acc_resultant"]
		timestamp = filtered_accelerations_df["timestamp"]

		# fig, axs = plt.subplots(1, 1, figsize=(6.4, 3), layout="constrained")
		# common to all three:
		# for ax in axs:

		# ax.plot("date", "adj_close", data=data)
		# plt.plot("timestamp", "SPEED", data=vel_df)
		for ax in axs:
			# Major ticks every half year, minor ticks every second,
			# ax.xaxis.set_major_locator(mdates.MinuteLocator(bysecond=range(60)))
			# ax.xaxis.set_minor_locator(mdates.MinuteLocator())
			ax.grid(True)
			
			# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
			# # Rotates and right-aligns the x labels so they don"t crowd each other.
			# for label in ax.get_xticklabels(which="major"):
			# 	label.set(rotation=30, horizontalalignment="right")

		# axs[0].scatter(timestamp, acc_x, s=0.1)
		axs[0].plot(timestamp, acc_x)
		axs[0].plot(timestamp, filtered_accelerations_df["filtered_acc_x"])
		# axs[0].legend(loc="right")
		axs[0].set_title("acc_x")

		axs[1].plot(timestamp, acc_y)
		axs[1].plot(timestamp, filtered_accelerations_df["filtered_acc_y"])
		axs[1].set_title("acc_y")

		axs[2].plot(timestamp, acc_z)    
		axs[2].plot(timestamp, filtered_accelerations_df["filtered_acc_z"])
		axs[2].set_title("acc_z")

		axs[3].plot(timestamp, acc_resultant)    
		axs[3].plot(timestamp, filtered_accelerations_df["acc_resultant"])
		axs[3].set_title("acc_resultant")

		for ax in axs:
			# ax.set_ylim(-10, 10)
			ax.set_xlim(lower_limit, upper_limit)

		# axs.legend(loc="center right", bbox_to_anchor=(1.25, 0.5))

		# fig.ylim(-0.2, 0.2)
		fig.tight_layout()

		imgs_path = os.path.join(self.root_dir, "images")
		path_exists = os.path.exists(imgs_path)
		if not path_exists:
			os.makedirs(imgs_path)

		fig_name = os.path.join(imgs_path, "fig_%02d" % img_id)
		fig.savefig(fig_name)

		plt.close(fig)

	def generate_graph_for_orientation(self):
		fig, axs = plt.subplots(ncols=1, nrows=3)
		orientation_df = self.orientation_df

		timestamp = orientation_df["timestamp"]
		azimuth = orientation_df["azimuth"]
		pitch = orientation_df["pitch"]
		roll = orientation_df["roll"]

		# min_timestamp = orientation_df["timestamp"].iloc[0]

		for ax in axs:
			ax.grid(True)

		# axs[0].scatter(timestamp, acc_x, s=0.1)
		axs[0].plot(timestamp, azimuth)
		axs[0].set_title("azimuth")

		axs[1].plot(timestamp, pitch)
		axs[1].set_title("pitch")

		axs[2].plot(timestamp, roll)    
		axs[2].set_title("roll")

		fig.tight_layout()
		plt.grid(True)

		# plt.scatter(timestamp, acc_x, s=0.1)
		plt.title("Orientation angle")

		# plt.show()
		return plt.gcf()

	def generate_graph_for_bearing(self):
		bearing_df = self.bearing_df

		# min_timestamp = bearing_df["timestamp"].iloc[0]

		timestamp = bearing_df["timestamp"]
		angle = bearing_df["angle"]

		plt.grid(True)

		# plt.scatter(timestamp, acc_x, s=0.1)
		plt.plot(timestamp, angle)
		plt.title("Bearing angle")

		# plt.show()
		return plt.gcf()


class UAHRideParser():
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.gps_df = self.create_gps_df()
		self.accelerometer_df = self.create_accelerometer_df()

	def create_gps_df(self):
		gps_file_path = os.path.join(self.root_dir, "RAW_GPS.txt")

		col_names = ["timestamp", "speed", "lat", "long", "altitude", "vert accuracy", "horiz accuracy", "course", "difcourse"] #, "?1", "?2", "?3", "?4"]

		return pd.read_csv(gps_file_path, sep=" ", names=col_names)

	def create_accelerometer_df(self):
		accelerometer_file_path = os.path.join(self.root_dir, "RAW_ACCELEROMETERS.txt")

		col_names = ["timestamp", "is speed gt 50 kmh", "acc_x", "acc_y", "acc_z", "filtered_acc_x", "filtered_acc_y", "filtered_acc_z", "roll_degrees", "pitch_degrees", "yaw_degrees", "?1", "?2", "?3", "?4"]
		acc_df = pd.read_csv(accelerometer_file_path, sep=" ", names=col_names)

		acc_df["acc_resultant"] = np.sqrt(acc_df["acc_x"] ** 2 + acc_df["acc_y"] ** 2 + acc_df["acc_z"] ** 2)

		return acc_df
	
	def generate_acc_sub_graph(self):
		for i in range(100):
			fig, axs = plt.subplots(ncols=1, nrows=4)

			accelerometer_df = self.accelerometer_df
			accelerometer_df = accelerometer_df[accelerometer_df.timestamp > i & accelerometer_df.timestamp < i + 20]
			acc_x = accelerometer_df["acc_x"]
			acc_y = accelerometer_df["acc_y"]
			acc_z = accelerometer_df["acc_z"]
			acc_resultant = accelerometer_df["acc_resultant"]
			timestamp = accelerometer_df["timestamp"]

			# axs[0].scatter(timestamp, acc_x, s=0.1)
			axs[0].plot(timestamp, accelerometer_df["acc_x"], label="acc_x")
			axs[0].plot(timestamp, accelerometer_df["filtered_acc_x"], label="filt_acc_x")
			axs[0].legend(loc="right")
			axs[0].set_title("acc_x")

			axs[1].plot(timestamp, acc_y)
			axs[1].plot(timestamp, accelerometer_df["filtered_acc_y"])
			axs[1].set_title("acc_y")

			axs[2].plot(timestamp, acc_z)
			axs[2].plot(timestamp, accelerometer_df["filtered_acc_z"])
			axs[2].set_title("acc_z")

			axs[3].plot(timestamp, acc_resultant)
			axs[3].plot(timestamp, accelerometer_df["acc_resultant"])
			axs[3].set_title("acc_resultant")

			

			# axs.legend(loc="center right", bbox_to_anchor=(1.25, 0.5))

			fig.tight_layout()

			# plt.show()

			img_buf = io.BytesIO()
			plt.savefig(img_buf, format="png")

			yield img_buf
				
	def foo(self, frame_granularity, delta_time_size, img_id):
		fig, axs = plt.subplots(ncols=1, nrows=3)

		accelerometer_df = self.accelerometer_df

		lower_limit = frame_granularity * img_id 
		upper_limit = frame_granularity * img_id + delta_time_size
		
		time_filter = (accelerometer_df.timestamp >= lower_limit) & (accelerometer_df.timestamp < upper_limit)
		filtered_accelerations_df = accelerometer_df[time_filter]

		acc_x = filtered_accelerations_df["acc_x"]
		acc_y = filtered_accelerations_df["acc_y"]
		acc_z = filtered_accelerations_df["acc_z"]
		timestamp = filtered_accelerations_df["timestamp"]

		axs[0].plot(timestamp, filtered_accelerations_df["acc_x"], label="acc_x")
		axs[0].plot(timestamp, filtered_accelerations_df["filtered_acc_x"], label="filt_acc_x")
		axs[0].legend(loc="right")
		axs[0].set_title("acc_x")

		axs[1].plot(timestamp, acc_y)
		axs[1].plot(timestamp, filtered_accelerations_df["filtered_acc_y"])
		axs[1].set_title("acc_y")

		axs[2].plot(timestamp, acc_z)    
		axs[2].plot(timestamp, filtered_accelerations_df["filtered_acc_z"])
		axs[2].set_title("acc_z")

		for ax in axs:
			ax.set_ylim(-0.2, 0.2)
			ax.set_xlim(lower_limit, upper_limit)

		fig.tight_layout()

		imgs_path = os.path.join(self.root_dir, "images")
		path_exists = os.path.exists(imgs_path)
		if not path_exists:
			os.makedirs(imgs_path)

		fig_name = os.path.join(imgs_path, "fig_%02d" % img_id)
		fig.savefig(fig_name)

		plt.close(fig)