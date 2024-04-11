
import subprocess
import os
import time
import json
import pandas as pd

class StatusInvestScraper():
	def __init__(self, root_save_folder="stock_data"):
		self.root_save_folder = root_save_folder

		if not os.path.exists(root_save_folder):
			os.makedirs(root_save_folder)

	def get_existing_stock_tickers(self):
		return ["bbse3", "itsa4", "cple6"]

	def get_all_available_data(self):
		stocks_names = self.get_existing_stock_tickers()

		for stock_ticker in stocks_names:
			stock_ticker_no_number = stock_ticker[:4]
			
			path_stock_csv_folder = os.path.join(self.root_save_folder, stock_ticker_no_number)
			path_stock_csv_file = os.path.join(path_stock_csv_folder, "data.csv")

			if os.path.exists(path_stock_csv_file):
				continue

			if not os.path.exists(path_stock_csv_folder):
				os.makedirs(path_stock_csv_folder)

			subprocess.check_call(["sh", "get_hist_data.sh", stock_ticker, "response.json"])

			stock_df = self.create_dataframe_from_json(stock_ticker)
			stock_df.to_csv(path_stock_csv_file)

			time.sleep(1)

	def create_dataframe_from_json(self, stock_ticker):
		json_content = open("response.json", "r").read()
		stock_dict = json.loads(json_content)

		stock_params = stock_dict["data"][stock_ticker]

		df_cols = ["param_name", "year", "value"]
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

		return pd.DataFrame(data=df_data_rows, columns=df_cols)

scraper = StatusInvestScraper()
scraper.get_all_available_data()