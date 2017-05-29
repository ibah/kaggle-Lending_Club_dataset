# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:33:55 2017
@author: msiwek
Kaggle dataset - Lending Club Loan Data
https://www.kaggle.com/wendykan/lending-club-loan-data
"""

import os
from sklearn.externals.joblib import dump, load
os.chdir('/home/michal/Dropbox/cooperation/_python/LendingClub-dataset/models')
os.chdir('D:\\data\\Dropbox\\cooperation\\_python\\LendingClub-dataset\\Models')
os.chdir('G:\\Dropbox\\cooperation\\_python\\LendingClub-dataset\\Models')

# Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, recall_score, precision_score
from sklearn.metrics import f1_score, fbeta_score
import warnings; warnings.filterwarnings('ignore')


#############################################

# preprocessing

#############################################


# Loading data
data = pd.read_csv('../input/loan.csv')

# Identifiers
set_0 = ['id', 'member_id', 'url']

# Credit application
set_1 = ['loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
         'emp_title', 'emp_length', 'home_ownership', 'annual_inc',
         'verification_status', 'desc', 'purpose', 'title', 'zip_code',
         'addr_state', 'dti', 'delinq_2yrs', 'earliest_cr_line',
         'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record',
         'open_acc', 'pub_rec', 'total_acc', 'initial_list_status',
         'collections_12_mths_ex_med', 'mths_since_last_major_derog',
         'policy_code', 'application_type', 'annual_inc_joint', 'dti_joint',
         'verification_status_joint', 'acc_now_delinq', 'tot_coll_amt',
         'tot_cur_bal', 'open_acc_6m', 'open_il_6m', 'open_il_12m',
         'open_il_24m', 'mths_since_rcnt_il',  'total_bal_il', 'il_util',
         'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util',
         'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m']

# Open credit
set_2 = ['funded_amnt', 'funded_amnt_inv', 'issue_d', 'pymnt_plan',
         'revol_bal', 'revol_util', # revol_bal maybe available during the application
         'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
         'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
         'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d',
         'last_credit_pull_d']

# Closed credit
set_3 = ['recoveries', 'collection_recovery_fee']

# Response variable
set_y = ['loan_status']


statuses = data.loan_status.unique()
default = np.array(['Charged Off', 'Default',
                   'Does not meet the credit policy. Status:Charged Off',
                   'Late (31-120 days)'])
repaid = np.array(['Fully Paid',
                   'Does not meet the credit policy. Status:Fully Paid'])
current = np.array(['Current', 'In Grace Period', 'Late (16-30 days)',
                    'Issued'])

# Helper functions to handle statuses
def make_binary(y):
    '''
    Converts status description into a binary indicator 1/0 (default/non-default).
    '''
    return pd.Series(y).apply(lambda x: 1 if x in default else 0)

def make_binary_label(y):
    '''
    Converts status description into a binary label (Default/Non-default).
    '''
    return pd.Series(y).apply(lambda x: 'Default' if x in default else 'Non-default')

# Ensuring missing data is properly coded
data.replace('n/a', np.nan, inplace=True)

# Features per type of preprocessing required

# drop
drop = ['id','member_id','url','emp_title','desc','title','zip_code']

# convert to datetime
date = ['earliest_cr_line', 'last_credit_pull_d', 'issue_d',
        'last_pymnt_d', 'next_pymnt_d']

# convert into dummies
dummy = ['addr_state', 'purpose', 'emp_length', 'grade',
         'home_ownership', 'verification_status_joint',
         'verification_status', 'pymnt_plan', 'application_type',
         'initial_list_status', 'term']

# apply calcuations
calculate = ['sub_grade']

# response variable
response = ['loan_status']

# numeric variables
numeric = [x for x in data.columns if x not in drop + date + dummy + calculate + response]

# Preprocessing function for transforming predictors into scikit-learn model inputs
def transform_predictors(raw_data):
    """
    Preprocessing for predictors
    
    Transforms predictors into a form suitable for scikit-learn models input:
    date variables -> pd.DateTime -> np.int64
    categorical variables -> dummies (one-hot encoding)
    sub_grade variable (A1, B2 etc.) -> numerical preserving ordering
    np.NaN -> -99
    
    Parameters
    ----------
    raw_data : pd.DataFrame
        any subset of of features/samples from the original dataset
    
    Returns
    -------
    pd.DataFrame
        predictors ready to be fed into scikit-learn models
    """
    
    # Date columns
    columns = np.intersect1d(date, raw_data.columns)
    date_columns = raw_data[columns].apply(
            pd.to_datetime, format='%b-%Y').astype(np.int64)
    date_columns = date_columns.apply(
            lambda column: column.apply(lambda x: -99 if x < 0 else x))
    
    # Categorical columns
    columns = np.intersect1d(dummy, raw_data.columns)
    dummy_columns = pd.get_dummies(raw_data[columns])
    
    # Calculated columns
    calculated_columns = raw_data.sub_grade.apply(
            lambda x: (ord(x[0])-65)*5+int(x[1]))
    columns = np.intersect1d(numeric, raw_data.columns)
    numeric_columns = raw_data[columns]
    X = pd.concat([date_columns, dummy_columns, calculated_columns,
                   numeric_columns], axis=1)
    
    # Dealing with missing values
    X.fillna(-99, inplace=True)
    
    return X

# Encoder from labels to integers
class CustomEncoder():
    """
    This is a wrapper for LabelEncoder so that it can be used in a pipeline
    """
    def __init__(self):
        self.le = LabelEncoder()
        
    def fit(self, y):
        self.le.fit(y)
        
    def transform(self, y):
        return self.le.transform(y).reshape(-1,1)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, y):
        return self.le.inverse_transform(y)

# Function returning multiclass y and a transformer pipeline
def make_multiclass(raw_y):
    """
    Returns
    -------
    np.array
        one-hot encoded response variable
    sklearn.pipeline.Pipeline
        a pipeline transforming multiple labels into one-hot encoding (and reverse)
    """
    response_pipe = make_pipeline(CustomEncoder(), LabelBinarizer())
    y_multi = response_pipe.fit_transform(raw_y)
    return y_multi, response_pipe

# Selecting relevant features and observations
sub_data = data[np.hstack([set_1, set_y])]
sub_data = sub_data[sub_data.loan_status.isin(np.hstack([default,repaid]))]

# select 10% of data
n_rows = sub_data.shape[0]
from scipy.stats import bernoulli
rows_selection = bernoulli.rvs(0.1, size=n_rows).astype('bool')
sub_data = sub_data[rows_selection]

# Preprocessing predictors and response
X = transform_predictors(sub_data)

# correct featuer names
import re
features = pd.Series(X.columns)
features = features.apply(lambda x: re.sub('[\[\]<]', '-', x))
X.columns = features

n_samples, n_features = X.shape
y_binary = make_binary(sub_data.loan_status)
y_multi, y_transformer = make_multiclass(sub_data.loan_status)

# Selecting target y
y = y_binary

###############################################################################

# Validation and Testing

###############################################################################


# Test-Train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
n_train = y_train.shape[0]
n_test = y_test.shape[0]

# Cross validation - will be used to evaluate the models using train data
skf = StratifiedKFold(n_splits=3, random_state=234)

# Reference Evaluation Measure
null_prediction = np.zeros(y_train.size)
null_accuracy = accuracy_score(y_train, null_prediction)
print("Null classifier accuracy score: %0.4f" % null_accuracy)


###############################################################################

# Model

###############################################################################


import xgboost as xgb

xgbc = xgb.XGBClassifier(
        max_depth=4,
        n_estimators=100)
xgbc.fit(X_train, y_train, verbose=True)

#############

# Evaluation

#############

# Evaluation
sub_rf_cv_pred = cross_val_predict(xgbc, X_train, y_train, cv=skf, n_jobs=2)
# Preformance measures
print(("Accuracy: %.4f\n" % accuracy_score(y_train, sub_rf_cv_pred)))
print(("Recall score: %.4f\n" % recall_score(y_train, sub_rf_cv_pred)))
print("Confusion matrix:")
print(confusion_matrix(y_train, sub_rf_cv_pred))
print("\nColumns: predicted non-default/default")
print("Rows: true non-default/default")


# Learning curve
train_sizes, train_scores, valid_scores = learning_curve(
        xgbc, X_train, y_train, cv=skf)

# Plotting the learning Curve
train_scores_mean = train_scores.mean(axis=1)
test_scores_mean = valid_scores.mean(axis=1)
plt.figure()
plt.ylim((.5,1))
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.axhline(y=null_accuracy, color='black', lw=1, linestyle='--', label="Null classifier score")
plt.title("Learning Curve", fontweight='bold')
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.legend(loc="best")
plt.show()

#############

# Test set

#############

# Predictions for the test set
pred = xgbc.predict(X_test)
pred_proba = xgbc.predict_proba(X_test)

# Preformance on the test set
print(("Accuracy: %.4f\n" % accuracy_score(y_test, pred)))
print(("Recall score: %.4f\n" % recall_score(y_test, pred)))
print("Confusion matrix:")
print(confusion_matrix(y_test, pred))
print("\nColumns: predicted non-default/default")
print("Rows: true non-default/default")

# ROC curve
y_score = pred_proba[:,1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic', fontweight='bold')
plt.legend(loc="lower right")
plt.show()














