import matplotlib.pyplot as plt
import pandas as pd

class StockDataPlotter:
	def __init__(self, df: pd.DataFrame):
		self.df = df

	def plot(self, param_name: str = "dy", tickers_to_plot: list[str] = None):
		if tickers_to_plot is None:
			tickers_to_plot = ["bbse3", "itsa4", "cple6"]

		param_df_filter = (self.df["param_name"] == param_name)

		ticker_df_filter = (self.df["ticker"] == None)
		for ticker in tickers_to_plot:
			ticker_df_filter = ( ticker_df_filter | (self.df["ticker"] == ticker) )
			
		complete_df_filter = (param_df_filter & ticker_df_filter)

		plt.title(param_name + " over time")

		for el in self.df[complete_df_filter].groupby("ticker"):
			legend_label = el[0]
			df_to_plot = el[1]

			x = df_to_plot["year"]
			y = df_to_plot["value"]

			plt.plot(x, y, label=legend_label)

		plt.legend()
		plt.show()
