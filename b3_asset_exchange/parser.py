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
			
		df = self.__standardize_col_names__(df)
		df = self.__standardize_table_data__(df)
		
		self.df = df

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
		return df
