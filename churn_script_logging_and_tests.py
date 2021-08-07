import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda):
	'''
	test perform eda function
	'''
	df = cls.import_data("./data/bank_data.csv")
	perform_eda(df)


def test_encoder_helper(encoder_helper):
	'''
	test encoder helper
	'''
	df = cls.import_data("./data/bank_data.csv")
	df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
	category_list = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
	df = encoder_helper(df, category_list, 'Churn')
	print(df.head())

def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''
	df = cls.import_data("./data/bank_data.csv")
	df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

def test_train_models(train_models):
	'''
	test train_models
	'''
	df = cls.import_data("./data/bank_data.csv")
	df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)


if __name__ == "__main__":
	#test_import(cls.import_data)
	#test_eda(cls.perform_eda)
	test_encoder_helper(cls.encoder_helper)
	#test_perform_feature_engineering(cls.perform_feature_engineering)
	#test_train_models(cls.train_models)





