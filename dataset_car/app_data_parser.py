import pandas as pd
import json
from datetime import datetime
import os

def generate_acc_from_vel(vel_df):
	n_samples = len(vel_df)

	data = []
	for i in range(n_samples - 1):
		time_1, vel_1 = tuple(vel_df.iloc[i])
		time_2, vel_2 = tuple(vel_df.iloc[i + 1])

		delta_t = time_2.timestamp() - time_1.timestamp()
		delta_v = (vel_2 - vel_1)

		if delta_t == 0:
			continue
		
		acc_kmh_p_second = delta_v / delta_t
		acc_m_p_second_2 = acc_kmh_p_second / 3.6
				
		data_line = {
			"timestamp": time_1,
			"SPEED": acc_m_p_second_2
		}

		data.append(data_line)

	acc_df = pd.DataFrame(data)

	return acc_df

def get_data_from_app():
	engine_data_path = os.path.join("RealData", "DELETEME.txt")

	# ex: Wed Jul 26 22:15:49 GMT+02:00 2023
	original_date_format = "%a %b %d %H:%M:%S GMT%z %Y"
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
		# timestamp = data_obj.timestamp() * 1000
		# print(timestamp)

		create_default_df = (lambda : pd.DataFrame(columns=["timestamp", param_name]))
		param_df = param_name_to_df.get(param_name, create_default_df())

		param_df.loc[len(param_df.index)] = [timestamp, param_value]
		param_name_to_df[param_name] = param_df

	speed_series = param_name_to_df["SPEED"]["SPEED"]
	convert_vel_to_float = lambda x : float(x.replace("km/h", ""))
	param_name_to_df["SPEED"]["SPEED"] = speed_series.apply(convert_vel_to_float)

	return param_name_to_df
