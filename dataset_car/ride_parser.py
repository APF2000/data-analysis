import os
import pandas as pd
import matplotlib.pyplot as plt

import io
import json
from datetime import datetime

# Fri Nov 03 13:37:58 GMT-03:00 2023
app_date_format = "%a %b %d %H:%M:%S %Z-03:00 %Y"

class RealRideParser():
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.gps_df = self.create_gps_df()
		# self.accelerometer_df = self.create_accelerometer_df()

	def create_gps_df(self):
		gps_file_path = os.path.join(self.root_dir, "DELETEME_GPS.txt")
		gps_file = open(gps_file_path, "r")

		data = []
		for line in gps_file.readlines():
			lat_long = json.loads(line[35:])
			original_time_string = line[:34]

			data_date = datetime.strptime(original_time_string, app_date_format)
			timestamp = data_date.timestamp()
			# print("date: ", data_date, timestamp)

			data_line = {
				"timestamp": timestamp,
				"original_time_string": original_time_string,
				"lat": float(lat_long[0]),
				"long": float(lat_long[1])
			}

			data.append(data_line)

		return pd.DataFrame(data)

	def create_accelerometer_df(self):
		accelerometer_file_path = os.path.join(self.root_dir, "RAW_ACCELEROMETERS.txt")

		col_names = ["timestamp", "is speed gt 50 kmh", "acc_x", "acc_y", "acc_z", "filtered_acc_x", "filtered_acc_y", "filtered_acc_z", "roll_degrees", "pitch_degrees", "yaw_degrees", "?1", "?2", "?3", "?4"]

		return pd.read_csv(accelerometer_file_path, sep=" ", names=col_names)

class UAHRideParser():
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.gps_df = self.create_gps_df()
		self.accelerometer_df = self.create_accelerometer_df()

	def create_gps_df(self):
		gps_file_path = os.path.join(self.root_dir, "RAW_GPS.txt")

		col_names = ["timestamp", "speed", "lat", "long", "altitude", "vert accuracy", "horiz accuracy", "course", "difcourse", "?1", "?2", "?3", "?4"]

		return pd.read_csv(gps_file_path, sep=" ", names=col_names)

	def create_accelerometer_df(self):
		accelerometer_file_path = os.path.join(self.root_dir, "RAW_ACCELEROMETERS.txt")

		col_names = ["timestamp", "is speed gt 50 kmh", "acc_x", "acc_y", "acc_z", "filtered_acc_x", "filtered_acc_y", "filtered_acc_z", "roll_degrees", "pitch_degrees", "yaw_degrees", "?1", "?2", "?3", "?4"]

		return pd.read_csv(accelerometer_file_path, sep=" ", names=col_names)
	
	def generate_acc_sub_graph(self):
		for i in range(100):
			fig, axs = plt.subplots(ncols=1, nrows=3)

			accelerometer_df = self.accelerometer_df
			accelerometer_df = accelerometer_df[accelerometer_df.timestamp > i & accelerometer_df.timestamp < i + 20]
			acc_x = accelerometer_df["acc_x"]
			acc_y = accelerometer_df["acc_y"]
			acc_z = accelerometer_df["acc_z"]
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

			# axs.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))

			fig.tight_layout()

			# plt.show()

			img_buf = io.BytesIO()
			plt.savefig(img_buf, format='png')

			yield img_buf