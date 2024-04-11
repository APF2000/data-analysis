
import subprocess
import os
import time
import json

import pandas as pd
import matplotlib.pyplot as plt

class StatusInvestScraper():
	def __init__(self, root_save_folder="stock_data"):
		self.root_save_folder = root_save_folder

		if not os.path.exists(root_save_folder):
			os.makedirs(root_save_folder)

	def get_existing_stock_tickers(self):
		return ["bbse3", "itsa4", "cple6"]

	def get_all_available_data(self):
		stocks_names = self.get_existing_stock_tickers()

		all_stocks_col_names = ["ticker", "param_name", "year", "value"]
		all_stocks_df = pd.DataFrame(columns=all_stocks_col_names)
		for stock_ticker in stocks_names:
			stock_ticker_no_number = stock_ticker[:4]

			path_stock_csv_folder = os.path.join(self.root_save_folder, stock_ticker_no_number)
			path_stock_csv_file = os.path.join(path_stock_csv_folder, "data.csv")

			if os.path.exists(path_stock_csv_file):
				stock_df = pd.read_csv(path_stock_csv_file)
			else:
				if not os.path.exists(path_stock_csv_folder):
					os.makedirs(path_stock_csv_folder)

				subprocess.check_call(["sh", "get_hist_data.sh", stock_ticker, "response.json"])

				stock_df = self.create_dataframe_from_json(stock_ticker)
				stock_df.to_csv(path_stock_csv_file, index=False)

			stock_df = stock_df.reindex(columns=all_stocks_col_names, fill_value=stock_ticker)
			all_stocks_df = pd.concat([all_stocks_df, stock_df], ignore_index=True, copy=False)

			time.sleep(1)

		return all_stocks_df

	def create_dataframe_from_json(self, stock_ticker):
		json_content = open("response.json", "r").read()
		stock_dict = json.loads(json_content)

		stock_params = stock_dict["data"][stock_ticker]

		df_col_names = ["param_name", "year", "value"]
		df_data_rows = []
		for param_dict in stock_params:
			param_name = param_dict["key"]
			param_historical_vals = param_dict["ranks"]

			for historical_val_dict in param_historical_vals:
				param_year = historical_val_dict["rank"]

				# raw_val = historical_val_dict["value_F"]
				# val_contains_number = any(char.isdigit() for char in raw_val)

				param_val = None
				if "value" in historical_val_dict.keys():
					param_val = historical_val_dict["value"]

				data_row = (param_name, param_year, param_val)
				df_data_rows.append(data_row)

		return pd.DataFrame(data=df_data_rows, columns=df_col_names)
	
class AssetDataPlotter():
	def __init__(self, df):
		self.df = df

	def plot(self, param_name, tickers_to_plot=["bbse3", "itsa4", "cple6"]):
		param_df_filter = (self.df["param_name"] == param_name)

		ticker_df_filter = (self.df["ticker"] == None)
		for ticker in tickers_to_plot:
			ticker_df_filter = ( ticker_df_filter | (self.df["ticker"] == ticker) )
			
		complete_df_filter = (param_df_filter & ticker_df_filter)

		filtered_df = self.df[complete_df_filter][["year", "value"]]
		filtered_df.plot(x="year", y="value")

		legend_labels = [el[0] for el in self.df[complete_df_filter].groupby("ticker")]
		dfs_to_plot = [el[1] for el in self.df[complete_df_filter].groupby("ticker")]


		for el in self.df[complete_df_filter].groupby("ticker"):
			legend_label = el[0]
			df_to_plot = el[1]

			x = df_to_plot["year"]
			y = df_to_plot["value"]

			plt.plot(x, y, label=legend_label)

		
		plt.legend()


		# x_data_series = [df["year"] for df in dfs_to_plot]
		# y_data_series = [df["value"] for df in dfs_to_plot]
		# labels_data_series = [df["ticker"] for df in dfs_to_plot]

		# legend_aux = plt.plot(x_data_series, y_data_series)
		# plt.legend(legend_aux, tickers_to_plot, loc=1)
		# plt.show()

		# plt.plot([x_data_series, y_data_series])
		# plt.legend(labels_data_series)

		plt.show()
  
# scraper = StatusInvestScraper()
# all_stocks_df = scraper.get_all_available_data()
# print(all_stocks_df)

# plotter = AssetDataPlotter(all_stocks_df)
# plotter.plot("dy")