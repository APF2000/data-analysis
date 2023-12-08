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

class RealRideParser():

	car_crimes_df = None
	lambda_url = "https://pntdpvkdsc.execute-api.us-east-1.amazonaws.com/default/app_data"

	def __init__(self, should_get_data_from_database=False, **params):

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

		# https://www.kaggle.com/code/jacekplonowski/sao-paulo-crime-eda/input?select=BO_2016.csv
		bo_2016_path = os.path.join("CrimeData", "BO_2016.csv")
		bo_2016_df = pd.read_csv(bo_2016_path)

		# https://www.kaggle.com/datasets/danlessa/geospatial-sao-paulo-crime-database/
		# https://www.kaggle.com/code/anagagodasilva/s-o-paulo-crime-maps-with-plotly/notebook
		sp_crimes_path = os.path.join("CrimeData", "crimes_por_bairro_sao_paulo.csv")

		sp_crimes_df = pd.read_csv(sp_crimes_path)
		RealRideParser.car_crimes_df = sp_crimes_df[sp_crimes_df["descricao"].str.contains("carro", na=False)]

	def calculate_crime_stats(self, map):
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
		
		# https://www.google.com.br/maps/place/S%C3%A9/@-23.5509876,-46.6345475,18.03z/data=!4m6!3m5!1s0x94ce59aa5b004689:0x37c720ec525c8bd9!8m2!3d-23.5500991!4d-46.633321!16s%2Fm%2F0g5583j?entry=ttu
		dangerous_place_lat_long = (-23.5509876,-46.6345475)

		dangerous_path_lat_longs = []
		for path_lat_long in self.gps_df[["lat", "long"]].iterrows():
			path_lat_long = tuple(path_lat_long[1])

			for _, reported_crime_lat_long in RealRideParser.car_crimes_df[["latitude", "longitude"]].iterrows():
				reported_crime_lat_long = tuple(reported_crime_lat_long)
				dist_from_target = get_dist_from_coords(path_lat_long, reported_crime_lat_long)

				# TODO: calcular proporcao de crimes num raio de 100 metros em comparação com um raio de 300m
				if dist_from_target <= 100:
					# print(reported_crime_lat_long, dist_from_target)
					dangerous_path_lat_longs.append(path_lat_long)

		for dangerous_latlong in dangerous_path_lat_longs:			
			folium.Marker(dangerous_latlong, icon=folium.Icon(icon="circle-exclamation", prefix="fa", color="red")).add_to(map)

		return map


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

		plt.show()

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

		plt.show()

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

		plt.show()

	def create_obd_df(self):
		if self.should_get_data_from_database:
			request_body = {
				"method": "get_obd_info",
				"data": {
					"user_id": self.user_id,
					"date_beg": self.date_beg,
					"date_end": self.date_end
				}
			}
			
			response = requests.post(RealRideParser.lambda_url, json=request_body)
			response_dict = json.loads(response.text)

			engine_data_list = []
			for data in response_dict["data"]:
				response_list = data[1:]
				corrected_list = ["" if x is None else x for x in response_list]
				engine_data = corrected_list[0] + " " + json.dumps([corrected_list[2], corrected_list[2], corrected_list[3]])
				engine_data_list.append(engine_data)
			engine_data = "\n".join(engine_data_list)
		else:
			engine_data_path = os.path.join(self.root_dir, "DELETEME.txt")
			engine_data = open(engine_data_path, "r").read().strip()

		param_name_to_df = {}
		for data_entry in engine_data.split("\n"):
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
			return float(rpm)
		
		param_name_to_df["SPEED"]["SPEED"] = speed_series.apply(convert_vel_to_float)
		param_name_to_df["ENGINE_RPM"]["ENGINE_RPM"] = param_name_to_df["ENGINE_RPM"]["ENGINE_RPM"].apply(convert_rpm_to_float)

		def convert_temp_to_float(x):
			temp = x.replace("C", "")
			if temp == "":
				return 0
			try:
				return float(temp)
			except:
				print("cannot convert %s to float" % temp)
				return 0

		def convert_pressure_to_float(x):
			pressure = x.replace("kPa", "")
			if pressure == "":
				return 0
			return float(pressure)

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
				"data": {
					"user_id": self.user_id,
					"date_beg": self.date_beg,
					"date_end": self.date_end
				}
			}
			
			response = requests.post(RealRideParser.lambda_url, json=request_body)
			response_dict = json.loads(response.text)

			gps_data_list = []
			for data in response_dict["data"]:
				response_list = data[1:]
				corrected_list = ["" if x is None else x for x in response_list]
				gps_data = corrected_list[0] + " " + json.dumps(corrected_list[2:])
				gps_data_list.append(gps_data)
			gps_data = "\n".join(gps_data_list)
		else:
			gps_file_path = os.path.join(self.root_dir, "DELETEME_GPS.txt")
			gps_data= open(gps_file_path, "r").read().strip()

		data = []

		for line in gps_data.split("\n"):

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
				"data": {
					"user_id": self.user_id,
					"date_beg": self.date_beg,
					"date_end": self.date_end
				}
			}
			
			response = requests.post(RealRideParser.lambda_url, json=request_body)
			response_dict = json.loads(response.text)

			acc_data_list = []
			for data in response_dict["data"]:
				response_list = data[1:]
				corrected_list = ["" if x is None else x for x in response_list]
				acc_data = corrected_list[0] + " " + json.dumps(corrected_list[2:])
				acc_data_list.append(acc_data)
			acc_data = "\n".join(acc_data_list)
		else:
			accelerometer_file_path = os.path.join(self.root_dir, "DELETEME_ACCELERATION.txt")
			acc_data= open(accelerometer_file_path, "r").read().strip()

		data = []

		for line in acc_data.split("\n"):

			parsed_data = self.parse_data_line(line)

			timestamp = parsed_data[0]
			acc_s = parsed_data[1:4]
			gravity_vec = parsed_data[4:7]

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
				"grav_z": gravity_vec[2]
			}

			data.append(data_line)

		acc_df = pd.DataFrame(data)

		acc_df["acc_resultant"] = np.sqrt(acc_df["acc_x"] ** 2 + acc_df["acc_y"] ** 2 + acc_df["acc_z"] ** 2)

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

		plt.show()

	def generate_graph_for_bearing(self):
		bearing_df = self.bearing_df

		# min_timestamp = bearing_df["timestamp"].iloc[0]

		timestamp = bearing_df["timestamp"]
		angle = bearing_df["angle"]

		plt.grid(True)

		# plt.scatter(timestamp, acc_x, s=0.1)
		plt.plot(timestamp, angle)
		plt.title("Bearing angle")

		plt.show()


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