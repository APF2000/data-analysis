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

		delta_t = time_2 - time_1
		delta_v = (vel_2 - vel_1)

		if delta_t == 0:
			continue
		
		acc_kmh_p_second = delta_v / delta_t
		acc_m_p_second_2 = acc_kmh_p_second / 3.6
				
		data_line = {
			"timestamp": time_1,
			"acceleration": acc_m_p_second_2
		}

		data.append(data_line)

	acc_df = pd.DataFrame(data)

	return acc_df

# https://en.wikipedia.org/wiki/Jerk_(physics)
def generate_jerk_from_acc(acc_df):
	n_samples = len(acc_df)

	data = []
	for i in range(n_samples - 1):
		time_1, vel_1 = tuple(acc_df.iloc[i])
		time_2, vel_2 = tuple(acc_df.iloc[i + 1])

		delta_t = time_2 - time_1
		delta_v = (vel_2 - vel_1)

		if delta_t == 0:
			continue
		
		acc_kmh_p_second = delta_v / delta_t
		acc_m_p_second_2 = acc_kmh_p_second / 3.6
				
		data_line = {
			"timestamp": time_1,
			"jerk": acc_m_p_second_2
		}

		data.append(data_line)

	jerk_df = pd.DataFrame(data)

	return jerk_df
