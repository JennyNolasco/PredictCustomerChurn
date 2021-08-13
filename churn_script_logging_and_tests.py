'''
This is file with unit test in best coding practices and score exceeding 7 when using pylint.

Author: Jeniffer Ferreira
Date: August 09, 2021
'''

import os
import logging
import math
import churn_library as cls
import constants as consts


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
        test data_import function 

        input:
                pth: a path to the csv
        output:
                df: pandas dataframe
        
        # Pass path where is the csv file
        ...
        # Check if file exists
        ...
        # Check if imported file has rows and columns
        ...
    '''
    try:
        df_raw = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df_raw.shape[0] > 0
        assert df_raw.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
        test perform_eda function 
        input:
                df: pandas dataframe

        output:
                None
        
        # Generate EDA 
        ...
        # Check if 5 image files geneated by EDA exists
        ...
    '''
    df_raw = cls.import_data("./data/bank_data.csv")

    df_raw['Churn'] = df_raw['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    perform_eda(df_raw)
    img_path = consts.IMG_EDA_PATH
    ext = '.png'
    png_files = [
        f for f in os.listdir(img_path) if os.path.isfile(
            os.path.join(img_path, f)) and f.endswith(ext)]

    try:
        assert len(png_files) == 5
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_eda: The eda images weren't found")
        raise err


def test_encoder_helper(encoder_helper):
    '''
        test encoder_helper function 

        input:
                df: pandas dataframe
                category_lst: list of columns that contain categorical features
                response: string of response name [optional argument that could be used for naming
                            variables or index y column]

        output:
                df: pandas dataframe with new columns for
        
        # Pass dataframe to create new columns and list with categorical features
        ...
        # Check if dataframe returned has all categorical columns
        ...
    '''
    df_raw = cls.import_data("./data/bank_data.csv")
    df_raw['Churn'] = df_raw['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    df_encoded = encoder_helper(df_raw, consts.CAT_COLUMNS, 'Churn')
    try:

        assert all(column in df_encoded.columns for column in consts.CATEGORICAL_VARS)
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The categorical vars weren't found")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
        test perform_feature_engineering function 

        input:
                df: pandas dataframe
                response: string of response name [optional argument that could be used for naming
                    variables or index y column]

        output:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        
        # Pass dataframe and column name for y column
        ...
        # Check if dataframes created have rows
        ...
        # Check if dataframes created have been separated correctly
        ...
    '''
    df_raw = cls.import_data("./data/bank_data.csv")
    df_raw['Churn'] = df_raw['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    df_encoded = cls.encoder_helper(df_raw, consts.CAT_COLUMNS, 'Churn')
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df_encoded, 'Churn')
    perc_30 = math.ceil((df_encoded.shape[0] * 30) / 100)
    perc_70 = math.floor((df_encoded.shape[0] * 70) / 100)

    try:
        assert X_test.shape[0] > 0
        assert X_train.shape[0] > 0
        assert y_test.shape[0] > 0
        assert y_train.shape[0] > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: dataframes don't appear to have rows")
        raise err

    try:
        assert X_test.shape[0] == perc_30
        assert X_train.shape[0] == perc_70
        assert y_test.shape[0] == perc_30
        assert y_train.shape[0] == perc_70
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: dataframes weren't divide correct")
        raise err


def test_train_models(train_models):
    '''
        test train_models function 

        input:
                y_train: training response values
                y_test:  test response values
                y_train_preds_lr: training predictions from logistic regression
                y_train_preds_rf: training predictions from random forest
                y_test_preds_lr: test predictions from logistic regression
                y_test_preds_rf: test predictions from random forest

        output:
                None
        
        # Pass dataframes 
        ...
        # Check if image files geneated by train model exists
        ...
        # Check if trained models exists
        ...
    '''
    df_raw = cls.import_data("./data/bank_data.csv")
    df_raw['Churn'] = df_raw['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    df_encoded = cls.encoder_helper(df_raw, consts.CAT_COLUMNS, 'Churn')
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
        df_encoded, 'Churn')
    train_models(X_train, X_test, y_train, y_test)

    img_path = consts.IMG_RESULTS_PATH
    ext = '.png'
    png_files = [
        f for f in os.listdir(img_path) if os.path.isfile(
            os.path.join(
                img_path, f)) and f.endswith(ext)]

    try:
        assert len(png_files) > 3
        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        logging.error("Testing train_models: The result images weren't found")
        raise err

    try:
        assert os.path.isfile(consts.RFC_MODEL_PATH)
        assert os.path.isfile(consts.LOGISTIC_MODEL_PATH)
        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        logging.error("Testing train_models: The trained models weren't found")
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
