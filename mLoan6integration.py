# -*- coding: utf-8 -*-
"""
Created on Tue May 16 20:48:54 2017
@author: a

Full report

Plan
Description of the dataset and the problem
The question
Plan
Raw exploration (types of data itp.)
Setting types and missing values
Visual exploration
Preprocessing
Binary class models
Multi class models
Model selection
Test set: ROC, AUC, precision-recall
Discussion of the best model
Two extensions
1) minimal features subset
2) real features (i.e. available before actual defaulting)
"""

import os
os.chdir('/home/michal/Dropbox/cooperation/_python/LendingClub-dataset/models')
os.chdir('D:\\data\\Dropbox\\cooperation\\_python\\LendingClub-dataset\\Models')
os.chdir('G:\\Dropbox\\cooperation\\_python\\LendingClub-dataset\\Models')
# magic: supress warning, inline plots

'''
Kaggle: complete loan data for all loans issued through the 2007-2015, including the current loan status (Current, Late, Fully Paid, etc.) and latest payment information. The file containing loan data through the "present" contains complete loan data for all loans issued through the previous completed calendar quarter. Additional features include credit scores, number of finance inquiries, address including zip codes, and state, and collections among others. The file is a matrix of about 890 thousand observations and 75 variables.
Online platform connecting borrowers and lenders. 
'''

# imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict

## Loading data
#data = pd.read_csv('../input/loan.csv')
#
## Inspection
## keep it simple, no one is interested in your tedious discovering of the obvious
## then read the forma and find out the real predictors
#n = data.shape[0]
#data.isnull().sum().sort_values(ascending=False)[:10]/n
## many missing values

###########################################################################

# only real columns

# load data
data = pd.read_csv('../input/loan.csv')

# id info
set_0 = ['id', 'member_id', 'url']
# credit application
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
# current loan
set_2 = ['funded_amnt', 'funded_amnt_inv', 'issue_d', 'pymnt_plan',
         'revol_bal', 'revol_util', # revol_bal maybe available during the application
         'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
         'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
         'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d',
         'last_credit_pull_d']
# closed loan
set_3 = ['recoveries', 'collection_recovery_fee']
# response
set_y = ['loan_status']
statuses = data.loan_status.unique()
default = np.array(['Charged Off', 'Default',
                   'Does not meet the credit policy. Status:Charged Off',
                   'Late (31-120 days)'])
repaid = np.array(['Fully Paid',
                   'Does not meet the credit policy. Status:Fully Paid'])
current = np.array(['Current', 'In Grace Period', 'Late (16-30 days)',
                    'Issued'])


# Coding missing data

# code missing data as np.nan
data.replace('n/a', np.nan, inplace=True)


# Selecting relevant features

# select feature set
#issue_d = data.issue_d.apply(pd.to_datetime, format='%b-%Y')
data = data[np.hstack([set_1, set_y])]


# Selecting relevant samples
# select time span

#np.bincount(issue_d >= '2011-01-01')
#plt.hist(issue_d >= '2011-01-01')
#data = data[issue_d >= '2011-01-01']
#data.shape
# (866565, 52)

# select rows
data = data[data.loan_status.isin(np.hstack([default,repaid]))]
data.shape
# (268530, 52)
# (247725, 52) after time adjustment







# Feature preprocessing (per type)

# Features per type of preprocessing required
drop = ['url','emp_title','desc','title','id','member_id','zip_code']
# title is represented by purpose
# zip_code requires to much memory, represented by addr_state (state)
# possibly you could aggregate zip_codes that perform well/bad in the training set
# and create an additional variables
date = ['earliest_cr_line', 'last_credit_pull_d', 'issue_d',
        'last_pymnt_d', 'next_pymnt_d']
dummy = ['addr_state', 'purpose', 'emp_length', 'grade',
         'home_ownership', 'verification_status_joint',
         'verification_status', 'pymnt_plan', 'application_type',
         'initial_list_status', 'term']
calculate = ['sub_grade']



# preprocess predictors

# Date columns
columns = np.intersect1d(date, data.columns)
date_columns = data[columns].apply(
                pd.to_datetime, format='%b-%Y').astype(np.int64)
date_columns = date_columns.apply(
        lambda column: column.apply(lambda x: -99 if x < 0 else x))
# Categorical columns
columns = np.intersect1d(dummy, data.columns)
dummy_columns = pd.get_dummies(data[columns])
# Calculated columns
calculated_columns = data.sub_grade.apply(
        lambda x: (ord(x[0])-65)*5+int(x[1]))

# Drop not needed columns
columns = np.intersect1d(np.hstack([drop, date, dummy, calculate]),
                         data.columns)
data.drop(columns, axis=1, inplace=True)

####################################################

# Response variable

####################################################

# 2-class version of y
def make_binary(y):
    return pd.Series(y).apply(lambda x: 1 if x in default else 0)

y_binary = make_binary(data.loan_status)

# 10-class version of y
#le = LabelEncoder()
#tmp = le.fit_transform(data.loan_status).reshape(-1,1)
#enc = OneHotEncoder()
#y_full = enc.fit_transform(tmp) # compressed sparse matrix
##print(y_full[:10,:10])
#from sklearn.preprocessing import label_binarize
#lb_y_full = label_binarize(tmp, range(10)) # numpy array, but the same content
##lb_y_full.shape

class CustomEncoder():
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
        return self.le.inverse_transform(y) #.reshape(-1,1)

# the same in a pipeline
response_pipe = make_pipeline(CustomEncoder(), LabelBinarizer())
y_multi = response_pipe.fit_transform(data.loan_status)
# response_pipe.inverse_transform(y_multi)

# drop the original response variable
data.drop('loan_status', axis=1, inplace=True)

# Creating final X data frame
X = pd.concat([data, date_columns, dummy_columns], axis=1)
n_samples, n_features = X.shape
# 157 columns, 887379 rows
# 137 columns (set 1), 268530 samples

# Dealing with missing values
X.fillna(-99, inplace=True)

# Test - Train split

# Selecting target y
y = y_binary
# Test-Train split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123)
n_train = y_train.shape[0]
n_test = y_test.shape[0]

# Cross validation

skf = StratifiedKFold(n_splits=3, random_state=234)
# no need shuffle=True as X is already shuffled on splitting

# Reference Evaluation Measure

y_train.value_counts(normalize=True)
#0    0.78103
#1    0.21897
null_prediction = np.zeros(y_test.size)
confusion_matrix(y_test, null_prediction) / n_test
#array([[ 0.7807942,  0.       ],
#       [ 0.2192058,  0.       ]])
accuracy_score(y_test, null_prediction)
# 0.78079420052383963

# Random Forest

rf = RandomForestClassifier(oob_score=True,
                            random_state=345,
                            n_jobs=1)

rf_params = {
        'n_estimators': [100], # 100
        'min_samples_split': [20],# 20 (30)
        'min_samples_leaf': [1], # (0) 1
        'max_features': ['sqrt']
        }

gs = GridSearchCV(rf, rf_params, cv=skf, n_jobs=-1)

#rf = RandomForestClassifier(criterion='gini',
#                             n_estimators=100,
#                             min_samples_split=10,
#                             min_samples_leaf=1,
#                             max_features='auto',
#                             oob_score=True,
#                             random_state=345,
#                             n_jobs=-1)
'''
grid search: max_features (sqrt, log2, None)
http://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html
'''
#rf.fit(X_train, y_train)
# 10 -> 1 min
# 50 -> 2 min
gs.fit(X_train, y_train)
# 100 trees, sqrt feat -> 3 min

# Results

gs.best_params_
#{'max_features': 'auto',
# 'min_samples_leaf': 1,
# 'min_samples_split': 20,
# 'n_estimators': 100} 3 min

feature_importance = pd.DataFrame(
        gs.best_estimator_.feature_importances_, index=X_train.columns,
        columns=['Importance'])
feature_importance.sort_values(by='Importance', ascending=False)
# 100 trees
#int_rate                                   7.812417e-02
#dti                                        7.034281e-02
#annual_inc                                 6.267115e-02
#earliest_cr_line                           5.651197e-02
#installment                                5.606133e-02
#tot_cur_bal                                4.959095e-02
#total_rev_hi_lim                           4.759377e-02
#loan_amnt                                  4.651444e-02
#total_acc                                  4.591881e-02
#open_acc                                   3.773823e-02
#mths_since_last_delinq                     3.064118e-02
#inq_last_6mths                             1.966882e-02
#mths_since_last_major_derog                1.760241e-02
#tot_coll_amt                               1.582784e-02
#term_ 60 months                            1.457964e-02
#mths_since_last_record                     1.409923e-02
#term_ 36 months                            1.284264e-02
#delinq_2yrs                                1.157700e-02
#grade_A                                    1.064138e-02
#grade_E                                    8.302621e-03
#grade_D                                    7.814646e-03
#emp_length_10+ years                       6.992275e-03
#pub_rec                                    6.991929e-03
#grade_B                                    6.624645e-03
#purpose_debt_consolidation                 6.569421e-03
#addr_state_CA                              6.566664e-03
#addr_state_NY                              6.251124e-03
#addr_state_FL                              5.969481e-03
#home_ownership_RENT                        5.968936e-03
#grade_C                                    5.888511e-03

# GSCV
#int_rate                                   8.257547e-02
#dti                                        7.125116e-02
#annual_inc                                 6.181687e-02
#installment                                5.385104e-02
#earliest_cr_line                           5.359386e-02
#tot_cur_bal                                4.808039e-02
#total_rev_hi_lim                           4.646354e-02
#loan_amnt                                  4.365628e-02
#total_acc                                  4.232912e-02
#open_acc                                   3.436722e-02
#mths_since_last_delinq                     2.862189e-02
#inq_last_6mths                             1.805063e-02
#term_ 60 months                            1.790547e-02
#mths_since_last_major_derog                1.778626e-02
#tot_coll_amt                               1.660091e-02
#term_ 36 months                            1.636527e-02
#grade_A                                    1.495601e-02
#mths_since_last_record                     1.484620e-02
#grade_E                                    1.199376e-02
#delinq_2yrs                                1.146832e-02
#grade_D                                    8.981368e-03
#grade_B                                    8.956408e-03
#pub_rec                                    6.843222e-03
#grade_C                                    6.616477e-03
#grade_F                                    6.343306e-03
#emp_length_10+ years                       6.007848e-03
#addr_state_NY                              5.836201e-03
#addr_state_CA                              5.815456e-03
#home_ownership_RENT                        5.702253e-03
#home_ownership_MORTGAGE                    5.544008e-03


# Evaluation

# Out-of-fold generalization error
print("%.4f" % gs.best_estimator_.oob_score_)
# 50 -> 0.7808
# 100 -> 0.7831
# 100, GS -> 0.7844

# cross_val_score(rf, X_train, y_train, n_jobs=-1)
cross_val_score(gs.best_estimator_, X_train, y_train, n_jobs=-1)
# array([ 0.7836794 ,  0.78347192,  0.78307292])

diag_pred = cross_val_predict(rf, X_train, y_train, cv=skf, n_jobs=-1)

confusion_matrix(y_train, diag_pred)

correct = y_train == diag_pred
correct.sum()
#  147,258
errors = y_train != diag_pred
errors.sum()
# 40,713 a bit too much to inspect
X_train[errors][:2]
y_train[errors][:2]
# you have high variance


# Learning curve

train_sizes, train_scores, valid_scores = learning_curve(
        RandomForestClassifier(min_samples_split=20,
                               n_estimators=20, n_jobs=-1),
#        RandomForestClassifier(min_samples_split=10,
#                               n_estimators=20, n_jobs=-1),
        X_train, y_train, cv=skf)
train_scores.mean(axis=1) # .868 -> .866
valid_scores.mean(axis=1) # .778 -> .778
'''
# fix high variance (you can't have more data) <- big difference between both
# fix bias <- flat learning curves....
# so maybe drop some features?
# adding more trees helps so that solves the bias partially
Flat - it's either bias or it's toughness of the problem, you really can't predict
with the features at hand
Variance, yes, because we overfit well

Possible solutions from data:
- split into two policy regimes by issue_d and work only on the recent past
- try to get more info from data (e.g. member_id, title?)

Possible modeling solutions:
- class weight
- multiclass pred
- ensembling
'''

best_features = feature_importance.sort_values(
        by='Importance', ascending=False)
features_selected = best_features.index[:20].values

X_selected = X[features_selected]
X = X_selected




# Generalization error

y_pred = gs.predict(X_test)
y_pred_proba = gs.predict_proba(X_test)
confusion_matrix(y_test, y_pred)
# GS
#array([[61852,  1048],
#       [16379,  1280]])
# on selected features
#array([[60763,  2137],
#       [15682,  1977]])
confusion_matrix(y_test, y_pred_proba[:,1]>0.25)
# weak results
# GS
#array([[43359, 19541],
#       [ 6820, 10839]])

'''
Ideas
grid search cv on this
learning curve
multiclass directly with random forest
other models
'''









# Multiclass

# Test - Train split

# Selecting target y
y = y_multi
# Test-Train split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123)
n_train = y_train.shape[0]
n_test = y_test.shape[0]

# Cross validation

skf = StratifiedKFold(n_splits=3, random_state=234)
# no need shuffle=True as X is already shuffled on splitting


#rf = RandomForestClassifier(oob_score=True,
#                            random_state=345,
#                            n_jobs=1)
#
#rf_params = {
#        'n_estimators': [100], # 100
#        'min_samples_split': [20],# 20 (30)
#        'min_samples_leaf': [1], # (0) 1
#        'max_features': ['sqrt']
#        }
#
#gs = GridSearchCV(rf, rf_params, cv=skf, n_jobs=-1)

rf = RandomForestClassifier(n_estimators=10,
                            min_samples_split=10,
                            min_samples_leaf=1,
                            max_features='sqrt',
                            oob_score=True,
                            random_state=345,
                            n_jobs=-1)
'''
grid search: max_features (sqrt, log2, None)
http://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html
'''
response_pipe.inverse_transform(np.array([[1,0,0,0,0,0]])) # d
response_pipe.inverse_transform(np.array([[0,1,0,0,0,0]])) # d
response_pipe.inverse_transform(np.array([[0,0,1,0,0,0]])) # d
response_pipe.inverse_transform(np.array([[0,0,0,1,0,0]]))
response_pipe.inverse_transform(np.array([[0,0,0,0,1,0]]))
response_pipe.inverse_transform(np.array([[0,0,0,0,0,1]])) # d

# setting weight x10 to default observations
weights = 9 * y_train[:,[0,1,2,5]].sum(1) + 1


rf.fit(X_train, y_train, weights)
# 10 -> 1 min
# 50 -> 2 min
#gs.fit(X_train, y_train)
# 100 trees, sqrt feat -> 3 min

# Results
pred = rf.predict(X_test)
pred_binary = make_binary(response_pipe.inverse_transform(pred))
resp_binary = make_binary(response_pipe.inverse_transform(y_test))
# Confusion Matrix
confusion_matrix(resp_binary, pred_binary)
# 10 trees
#array([[59600,  3300],
#       [15180,  2479]]) # this is promising
#array([[46818, 16082],
#       [ 8999,  8660]]) # with weights



    
    
    



# LDA - very bad
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
# Fitting
lda.fit(X_train, y_train)
# Evaluation
lda.coef_
lda.intercept_
lda.explained_variance_ratio_
#lda_coef_ = np.argsort(lda.coef_)
# <------------------------------ select max+ and max- features
# Prediction
pred = lda.predict(X_test)
pred_proba = lda.predict_proba(X_test)
# Confusion Matrix
confusion_matrix(y_test, pred)
