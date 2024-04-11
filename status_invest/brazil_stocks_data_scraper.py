
import subprocess
import os
import time
import json

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
			path_stock_csv = os.path.join(self.root_save_folder, stock_ticker_no_number, "data.csv")

			if os.path.exists(path_stock_csv):
				continue

			subprocess.check_call(["sh", "get_hist_data.sh", stock_ticker, "response.json"])

			stock_df = self.create_dataframe_from_json(stock_ticker)

			time.sleep(1)

	def create_dataframe_from_json(self, stock_ticker):
		json_content = open("response.json", "r").read()
		stock_dict = json.loads(json_content)

		stock_params = stock_dict["data"][stock_ticker]
		for param_dict in stock_params:
			param_name = param_dict["key"]
			param_historical_vals = param_dict["ranks"]

	def save_scraped_stock_data(self, path_stock_csv, df_stock_data):

		df_stock_data.to_csv(path_stock_csv)

scraper = StatusInvestScraper()
scraper.get_all_available_data()