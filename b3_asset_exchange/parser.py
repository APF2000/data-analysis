import pandas as pd
import os

#%pip install openpyxl --upgrade

class B3Parser():
	def __init__(self, relative_path, file_type='excel'):
		base_path = os.path.dirname(os.path.abspath(__file__))
		print('base_path', base_path)
		
		full_path = os.path.join(base_path, relative_path)
		
	
		if file_type == 'excel':
			df = pd.read_excel(full_path)
		elif file_type == 'csv':
			df = pd.read_csv(full_path)
		else:
			print(f'unsupported file type {file_type}')
			return
			
		self.raw_df = df
			
		df = self.__standardize_col_names__(df)
		df = self.__standardize_table_data__(df)
		
		final_df = df
		
		self.buy_and_sell_df = self.__get_buy_and_sell_df__(final_df)
		self.updates_df = self.__get_updates_df__(final_df)
		
		
		self.final_df = final_df

	def __standardize_col_names__(self, df):
		col_name_translator = {
			'Entrada/Saída': 'debit_or_credit',
			'Data': 'op_date',
			'Movimentação': 'financial_movement_type',
			'Produto': 'product',
			'Instituição': 'broker_name',
			'Quantidade': 'qtty',
			'Preço unitário': 'unit_price',
			'Valor da Operação': 'op_total_amount'
		}
		
		return df.rename(columns=col_name_translator)

	def __standardize_table_data__(self, df):
		df = df.replace({
			'debit_or_credit': {
				'Debito': 'debit',
				'Credito': 'credit'
			}
		})
		
		df['op_date'] = pd.to_datetime(df['op_date'], format='%d/%m/%Y')
		
		df = df.replace({
			'financial_movement_type': {
				'Transferência - Liquidação': 'buy_and_sell',
				'COMPRA/VENDA': 'buy_and_sell',
				'PAGAMENTO DE JUROS': 'dividends',
				'Rendimento': 'dividends',
				'Juros Sobre Capital Próprio': 'dividends',
				'Dividendo': 'dividends',
				'Empréstimo': 'asset_rent',
				'Atualização': 'asset_update'
			}
		})
		
			# 'Produto': 'product',
			# 'Instituição': 'broker_name',
			# 'Quantidade': 'qtty',
			# 'Preço unitário': 'unit_price',
			# 'Valor da Operação': 'op_total_amount'
			
		df['ticker_code'] = df['product'].apply(lambda x: x.split('-')[0].strip())
		df['legal_name'] = df['product'].apply(lambda x: x.split('-')[1].strip())
			
		return df
		
	def __get_buy_and_sell_df__(self, final_df):
		filter_series = (final_df['financial_movement_type'] == 'buy_and_sell')
		
		return final_df[filter_series]
		
	def __get_updates_df__(self, final_df):
		filter_series = (final_df['financial_movement_type'] == 'asset_update')
		
		return final_df[filter_series]
		
