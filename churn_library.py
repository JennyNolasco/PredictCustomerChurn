'''
This is a project in best coding practices and score exceeding 7 when using pylint.
Using sklearn library to predict

Author: Jeniffer Ferreira
Date: August 09, 2021
'''

# import libraries
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import constants as consts
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    return pd.read_csv(pth, index_col=0)


def perform_eda(df_raw):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    plt.figure(figsize=(20, 10))
    df_raw['Churn'].hist()
    plt.savefig(f'{consts.IMG_EDA_PATH}Churn_hist')

    plt.figure(figsize=(20, 10))
    df_raw['Customer_Age'].hist()
    plt.savefig(f'{consts.IMG_EDA_PATH}Customer_Age_hist')

    plt.figure(figsize=(20, 10))
    df_raw.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(f'{consts.IMG_EDA_PATH}Marital_Status_bar')

    plt.figure(figsize=(20, 10))
    sns.distplot(df_raw['Total_Trans_Ct'])
    plt.savefig(f'{consts.IMG_EDA_PATH}Total_Trans_Ct_distplot')

    plt.figure(figsize=(20, 10))
    sns.heatmap(df_raw.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(f'{consts.IMG_EDA_PATH}Corr_dark2_r')


def create_encoded_column_lst(df_raw, column_name):
    '''
    create_category_lst function to a list with categorical column into

    input:
            df: pandas dataframe
            column_name: str categorical column name

    output:
            categorical_list: list
    '''
    encoded_lst = []
    encoded_groups = df_raw.groupby(column_name).mean()['Churn']

    for val in df_raw[column_name]:
        encoded_lst.append(encoded_groups.loc[val])

    return encoded_lst


def encoder_helper(df_raw, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming
                        variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for column_name in category_lst:
        encoded_lst = create_encoded_column_lst(df_raw, column_name)
        name_new_column = f'{column_name}_{response}'

        df_raw[name_new_column] = encoded_lst

    return df_raw


def perform_feature_engineering(df_encoded, response):
    '''
    input:
            df: pandas dataframe
            response: string of response name [optional argument that could be used for naming
                variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    X = pd.DataFrame()
    X[consts.CATEGORICAL_VARS] = df_encoded[consts.CATEGORICAL_VARS]

    y = df_encoded[response]
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    # scores
    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.show()
    plt.savefig(f'{consts.IMG_RESULTS_PATH}classification_report_RF', dpi=100)

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(f'{consts.IMG_RESULTS_PATH}classification_report_LR', dpi=100)


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    plt.figure(figsize=(15, 8))
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.savefig(f'{consts.IMG_RESULTS_PATH}lrc_plot')

    plt.figure(figsize=(15, 8))
    roc_ax = plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc, X_test, y_test, ax=roc_ax, alpha=0.8)
    lrc_plot.plot(ax=roc_ax, alpha=0.8)
    plt.savefig(f'{consts.IMG_RESULTS_PATH}lrc_rfc_plot')

    plt.figure(figsize=(15, 8))
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    plt.savefig(f'{consts.IMG_RESULTS_PATH}summary_plot')

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, consts.RFC_MODEL_PATH)
    joblib.dump(lrc, consts.LOGISTIC_MODEL_PATH)
