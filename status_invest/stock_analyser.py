import pandas as pd
import re

def std_col_name(name: str) -> str:
	name = name.strip()
	name = re.sub(r"\s+", "_", name)
	name = re.sub(r"[^$/\w]", "", name)
	return name.lower()

def percentage_str_to_float(x: str) -> float:
	x = str(x)
	x = re.sub(r"^\s*$", "N/A", x)
	x = x.strip("%").replace(",", ".")
	if x == "N/A":
		return None
	return float(x) / 100

class StockAnalyzer:
	def __init__(self, stocks_file: str):
		self.stocks_file = stocks_file
		self.stocks_df = self._load_and_clean_stocks()

	def _load_and_clean_stocks(self) -> pd.DataFrame:
		# Carrega e limpa os dados de ações
		br_stocks = pd.read_csv(self.stocks_file, sep=";", decimal=",", thousands='.')
		stocks_col_names = [std_col_name(name) for name in br_stocks.columns]
		br_stocks.columns = stocks_col_names
		br_stocks[["dy"]] = br_stocks[["dy"]].fillna(0)
		return br_stocks

	def apply_green_flag_filters(self, feature_to_green_flag_filter: dict = None) -> pd.DataFrame:
		if feature_to_green_flag_filter is None:
			feature_to_green_flag_filter = {
				"dy": {"min_val": 0.06, "max_val": float('inf')},
				"roe": {"min_val": 0.10, "max_val": float('inf')},
				"roa": {"min_val": 0.05, "max_val": float('inf')},
				"margem_bruta": {"min_val": 0.40, "max_val": float('inf')},
				"margem_liquida": {"min_val": 0.10, "max_val": float('inf')},
				"liquidez_corrente": {"min_val": 1.0, "max_val": float('inf')},
				"p_vp": {"min_val": 0.0, "max_val": 1.0}
			}

		filtered_br_stocks = self.stocks_df
		for feature_name, filter_info in feature_to_green_flag_filter.items():
			min_val = filter_info["min_val"]
			max_val = filter_info["max_val"]
			min_filter = (filtered_br_stocks[feature_name] > min_val)
			max_filter = (filtered_br_stocks[feature_name] < max_val)
			filtered_br_stocks = filtered_br_stocks[min_filter & max_filter]
		return filtered_br_stocks

	
	def generate_points(self, feature_to_green_flag_filter: dict = None, feature_to_red_flag_filter: dict = None) -> pd.DataFrame:
		if feature_to_green_flag_filter is None:
			feature_to_green_flag_filter = {
				"dy": {"min_val": 0.06, "max_val": float('inf')},
				"roe": {"min_val": 0.10, "max_val": float('inf')},
				"roa": {"min_val": 0.05, "max_val": float('inf')},
				"margem_bruta": {"min_val": 0.40, "max_val": float('inf')},
				"margem_liquida": {"min_val": 0.10, "max_val": float('inf')},
				"liquidez_corrente": {"min_val": 1.0, "max_val": float('inf')},
				"p_vp": {"min_val": 0.0, "max_val": 1.0}
			}

		if feature_to_red_flag_filter is None:
			feature_to_red_flag_filter = {
				"divida_bruta": {"min_val": 0.5, "max_val": float('inf')}
			}

		df_points = pd.DataFrame()
		df_points_generic = pd.DataFrame()
		feature_col_names = self.stocks_df.columns[2:]

		for col_name in feature_col_names:
			if col_name not in feature_to_green_flag_filter or col_name not in feature_to_red_flag_filter:
				continue

			filter_dict = feature_to_green_flag_filter[col_name]
			red_flag_filters = feature_to_red_flag_filter[col_name]

			df_points_generic[["generic_point_" + col_name]] = self._generate_point_col(self.stocks_df, col_name)
			df_points[["green_point_" + col_name, "red_point_" + col_name]] = self._generate_point_col_with_rank(self.stocks_df, col_name, filter_dict, red_flag_filters)

		df_points["ticker"] = self.stocks_df["ticker"]
		df_points["total_points"] = df_points.iloc[:, :-1].sum(axis=1)
		df_points_generic["total_generic_points"] = df_points_generic.iloc[:, :-1].sum(axis=1)

		df_all_points = pd.merge(df_points, df_points_generic, left_index=True, right_index=True, suffixes=[None, "_deleteme"])
		df_all_points = pd.merge(df_all_points, self.stocks_df, left_index=True, right_index=True, suffixes=[None, "_deleteme"])

		df_all_points["p/l"] = self.stocks_df["p/l"]
		df_all_points["dy"] = self.stocks_df["dy"]

		stocks_rank = df_all_points.sort_values(by=["total_points", "dy", "total_generic_points"], ascending=False)
		return stocks_rank.head(20)[["ticker", "dy", "p/l", "total_points", "total_generic_points"]]

	def _generate_point_col(self, stock_df: pd.DataFrame, col_name: str) -> pd.DataFrame:
		# Aqui vai a implementação do método generate_point_col
		# (Reaproveitar o código existente para esta função)
		pass

	def _generate_point_col_with_rank(self, stock_df: pd.DataFrame, feature_name: str, filter_dict: dict, red_flag_filters: dict) -> pd.DataFrame:
		# Aqui vai a implementação do método generate_point_col_with_rank
		# (Reaproveitar o código existente para esta função)
		pass

class FundAnalyzer:
	def __init__(self, funds_file: str):
		self.funds_file = funds_file
		self.funds_df = self._load_and_clean_funds()

	def _load_and_clean_funds(self) -> pd.DataFrame:
		# Carrega e limpa os dados de fundos
		fiis_df = pd.read_csv(self.funds_file, sep="\t", decimal=",", thousands=".")
		fiis_col_names = [std_col_name(name) for name in fiis_df.columns]
		fiis_df.columns = fiis_col_names
		fiis_df.rename(columns={"dividend_yield": "dy", "fundos": "ticker"}, inplace=True)
		fiis_df[["dy"]] = fiis_df[["dy"]].fillna(0)
		return fiis_df

	def generate_fund_points(self, feature_to_green_flag_filter: dict, feature_to_red_flag_filter: dict) -> pd.DataFrame:
		# Implementação semelhante ao método generate_points no StockAnalyzer
		pass

	# Implementar _generate_point_col e _generate_point_col_with_rank de forma similar

class PortfolioAnalyzer:
	def __init__(self, stock_analyzer: StockAnalyzer, fund_analyzer: FundAnalyzer):
		self.stock_analyzer = stock_analyzer
		self.fund_analyzer = fund_analyzer

	def analyze_portfolio(self, stock_filters: dict, fund_filters: dict, red_flag_filters: dict) -> pd.DataFrame:
		stocks_rank = self.stock_analyzer.generate_points(stock_filters, red_flag_filters)
		funds_rank = self.fund_analyzer.generate_fund_points(fund_filters, red_flag_filters)
		portfolio = pd.concat([stocks_rank, funds_rank], ignore_index=True)
		return portfolio

