import pandas as pd

class StockDataAnalyser:
	def __init__(self, stock_data: pd.DataFrame):
		self.stock_data = stock_data

	def analyse(self) -> pd.DataFrame:
		# Exemplo de join ou outra operação de análise
		# Para este exemplo, podemos fazer um merge entre diferentes dataframes
		# Considerando que você terá múltiplas operações aqui.
		analysed_df = self.stock_data.copy()
		# Realizar operações de análise e cálculo aqui.
		# ...
		return analysed_df

	def calculate_score(self, df: pd.DataFrame) -> pd.DataFrame:
		# Exemplo de cálculo de score baseado em flags
		df['score'] = 0  # Inicializar com 0
		# Aplicar as regras para calcular o score com base em flags
		# Exemplo:
		df.loc[df['param_name'] == 'green_flag', 'score'] += 1
		df.loc[df['param_name'] == 'red_flag', 'score'] -= 1
		return df
