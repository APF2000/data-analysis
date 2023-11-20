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


# Fri Nov 03 13:37:58 GMT-03:00 2023
app_date_format = "%a %b %d %H:%M:%S %Z%z %Y"

class RealRideParser():
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.gps_df = self.create_gps_df()
		self.accelerometer_df = self.create_accelerometer_df()
		self.obd_data = self.get_data_from_app()

	def get_data_from_app(self):
		engine_data_path = os.path.join(self.root_dir, "DELETEME.txt")

		# ex: Wed Jul 26 22:15:49 GMT+02:00 2023
		original_date_format = "%a %b %d %H:%M:%S %Z%z %Y"
		# desired_date_format = "%Y-%m-%d %H:%M:%S"

		param_name_to_df = {}

		engine_data = open(engine_data_path, "r").read().strip()
		for data_entry in engine_data.split("\n"):
			data_entry = data_entry.replace("NODATA", "0")

			date = data_entry[:34]
			info_list = json.loads(data_entry[35:])

			param_name = info_list[0]
			param_value = info_list[2]

			# print("date: ", date)
			# print("info_list: ", info_list)

			data_obj = datetime.strptime(date, original_date_format)
			timestamp = data_obj
			timestamp = data_obj.timestamp() # * 1000
			# print(timestamp)

			create_default_df = (lambda : pd.DataFrame(columns=["timestamp", param_name]))
			param_df = param_name_to_df.get(param_name, create_default_df())

			param_df.loc[len(param_df.index)] = [timestamp, param_value]
			param_name_to_df[param_name] = param_df

		speed_series = param_name_to_df["SPEED"]["SPEED"]
		convert_vel_to_float = lambda x : float(x.replace("km/h", ""))
		param_name_to_df["SPEED"]["SPEED"] = speed_series.apply(convert_vel_to_float)

		return param_name_to_df

	def create_velocity_from_gps_df(self):
		gps_df = self.gps_df #.drop_duplicates(subset=["timestamp"])
		n_samples = len(gps_df)

		data = []
		desired_cols = ["timestamp", "lat", "long"]
		for i in range(n_samples - 1):
			time_1, lat_1, long_1 = tuple(gps_df[desired_cols].iloc[i])
			time_2, lat_2, long_2 = tuple(gps_df[desired_cols].iloc[i + 1])

			# time_1 = time_1.timestamp()
			# time_2 = time_2.timestamp()

			# print("time_1", time_1)
			# print("time_2", time_2)

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

			# x1, y1 = pyproj.transform("wgs84", "epsg3035", long_1, lat_1)
			# print(x1, y1)

			# x2, y2 = pyproj.transform("wgs84", "epsg3035", long_2, lat_2)
			# print(x2, y2)

			# # a Pythagore's theorem is sufficient to compute an approximate distance
			# distance_m = np.sqrt((x2-x1)**2 + (y2-y1)**2)
			# print(distance_m)

			data.append(data_line)

		velocity_df = pd.DataFrame(data)

		return velocity_df

	def get_acc_stats(self):
		acc_df = self.accelerometer_df

		mean_acc = acc_df["acc_resultant"].mean()
		std_dev_acc = acc_df["acc_resultant"].std()
		median = acc_df["acc_resultant"].median()

		stats = {
			"mean_acc": mean_acc,
			"std_dev_acc": std_dev_acc,
			"median": median
		}

		return acc_df["acc_resultant"].describe() #tats
	
	def parse_data_line(self, line):
		actual_data = json.loads(line[35:])
		dot_convert_foo = lambda x: float(x.replace(',', '.'))
		actual_data = list(map(dot_convert_foo, actual_data))
		original_time_string = line[:34]

		data_date = datetime.strptime(original_time_string, app_date_format)
		timestamp = data_date.timestamp()

		parsed_data_list = [timestamp] + actual_data
		return tuple(parsed_data_list)


	def create_gps_df(self):
		gps_file_path = os.path.join(self.root_dir, "DELETEME_GPS.txt")
		gps_file = open(gps_file_path, "r")

		data = []
		for line in gps_file.readlines():

			timestamp, lat, long = self.parse_data_line(line)

			data_line = {
				"timestamp": timestamp,
				"lat": float(lat),
				"long": float(long)
			}

			data.append(data_line)

		return pd.DataFrame(data)

	def create_accelerometer_df(self):
		accelerometer_file_path = os.path.join(self.root_dir, "DELETEME_ACCELERATION.txt")

		gps_file = open(accelerometer_file_path, "r")

		data = []
		for line in gps_file.readlines():

			parsed_data = self.parse_data_line(line)

			timestamp = parsed_data[0]
			acc_s = parsed_data[1:4]

			data_line = {
				"timestamp": timestamp,
				"acc_x": acc_s[0],
				"acc_y": acc_s[1],
				"acc_z": acc_s[2],
				"filtered_acc_x": acc_s[0],
				"filtered_acc_y": acc_s[1],
				"filtered_acc_z": acc_s[2]
			}

			data.append(data_line)

		acc_df = pd.DataFrame(data)

		acc_df["acc_resultant"] = np.sqrt(acc_df["acc_x"] ** 2 + acc_df["acc_y"] ** 2 + acc_df["acc_z"] ** 2)

		return acc_df

	def foo_for_real_data(self, frame_granularity, delta_time_size, img_id):
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

		# fig, axs = plt.subplots(1, 1, figsize=(6.4, 3), layout='constrained')
		# common to all three:
		# for ax in axs:

		# ax.plot('date', 'adj_close', data=data)
		# plt.plot("timestamp", "SPEED", data=vel_df)
		for ax in axs:
			# Major ticks every half year, minor ticks every second,
			# ax.xaxis.set_major_locator(mdates.MinuteLocator(bysecond=range(60)))
			# ax.xaxis.set_minor_locator(mdates.MinuteLocator())
			ax.grid(True)
			
			# ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
			# # Rotates and right-aligns the x labels so they don't crowd each other.
			# for label in ax.get_xticklabels(which='major'):
			# 	label.set(rotation=30, horizontalalignment='right')

		# axs[0].scatter(timestamp, acc_x, s=0.1)
		axs[0].plot(timestamp, acc_x)
		axs[0].plot(timestamp, filtered_accelerations_df["filtered_acc_x"])
		# axs[0].legend(loc='right')
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

		# axs.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))

		# fig.ylim(-0.2, 0.2)
		fig.tight_layout()

		imgs_path = os.path.join(self.root_dir, "images")
		path_exists = os.path.exists(imgs_path)
		if not path_exists:
			os.makedirs(imgs_path)

		fig_name = os.path.join(imgs_path, "fig_%02d" % img_id)
		fig.savefig(fig_name)

		plt.close(fig)


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
			axs[0].legend(loc='right')
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

			

			# axs.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))

			fig.tight_layout()

			# plt.show()

			img_buf = io.BytesIO()
			plt.savefig(img_buf, format='png')

			yield img_buf
				
	def foo(self, frame_granularity, delta_time_size, img_id):
		fig, axs = plt.subplots(ncols=1, nrows=3)

		accelerometer_df = self.accelerometer_df

		lower_limit = frame_granularity * img_id 
		upper_limit = frame_granularity * img_id + delta_time_size
		# print("low up", lower_limit, upper_limit)
		time_filter = (accelerometer_df.timestamp >= lower_limit) & (accelerometer_df.timestamp < upper_limit)
		filtered_accelerations_df = accelerometer_df[time_filter]

		acc_x = filtered_accelerations_df["acc_x"]
		acc_y = filtered_accelerations_df["acc_y"]
		acc_z = filtered_accelerations_df["acc_z"]
		timestamp = filtered_accelerations_df["timestamp"]

		# axs[0].scatter(timestamp, acc_x, s=0.1)
		axs[0].plot(timestamp, filtered_accelerations_df["acc_x"], label="acc_x")
		axs[0].plot(timestamp, filtered_accelerations_df["filtered_acc_x"], label="filt_acc_x")
		axs[0].legend(loc='right')
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

		# axs.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))

		# fig.ylim(-0.2, 0.2)
		fig.tight_layout()

		imgs_path = os.path.join(self.root_dir, "images")
		path_exists = os.path.exists(imgs_path)
		if not path_exists:
			os.makedirs(imgs_path)

		fig_name = os.path.join(imgs_path, "fig_%02d" % img_id)
		fig.savefig(fig_name)

		plt.close(fig)