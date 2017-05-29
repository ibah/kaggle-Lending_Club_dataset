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

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#############################################

# preprocessing

#############################################

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
# from sklearn.pipeline import make_pipeline

# load data
data = pd.read_csv('../input/loan.csv')
# recognized missing data
data.replace('n/a', np.nan, inplace=True)


#############################################

# Visual Exploration

#############################################

y = data.loan_amnt
plt.hist(y)
sns.kdeplot(y)
sns.distplot(y) # good
# -> shows focal points, round values are over-represented

x = data.issue_d.apply(pd.to_datetime, format='%b-%Y')
tmp = y.groupby(x).sum()

a = 'sjsklsjksjslkj'
a[:3]+'..'+a[-3:]

def shorten(x):
    if len(x) > 22:
        x = x[:6] + '..' + x[-11:]
    return x
v_shorten = np.vectorize(shorten)
v_shorten(a)

statuses = data.loan_status.unique()
default = np.array(['Charged Off',
                   'Default',
                   'Does not meet the credit policy. Status:Charged Off',
                   'Late (31-120 days)'])
def make_binary(y):
    return pd.Series(y).apply(lambda x: 1 if x in default else 0)


data.index = x
# value monthy
loan_amnt_monthly = data.loan_amnt.groupby(pd.TimeGrouper(freq='M')).sum()
loan_amnt_monthly.plot()
# cound monthly
loan_issued_monthly = data.loan_amnt.groupby(pd.TimeGrouper(freq='M')).count()
loan_issued_monthly.plot()
# average loan value
loan_amnt_monthly.divide(loan_issued_monthly).plot()

# amount per status - beautiful!
loan_amnt_per_status = data.groupby('loan_status').loan_amnt.sum()
#loan_amnt_per_status.plot.bar()
#plt.bar(range(10), np.log1p(loan_amnt_per_status), color=color)
#plt.xticks(range(10), v_shorten(loan_amnt_per_status.index), rotation=45)
color = make_binary(loan_amnt_per_status.index).map({1:'C2', 0:'C0'})
fig, ax = plt.subplots(1)
plt.bar(range(10), np.log1p(loan_amnt_per_status), color=color,
        tick_label=loan_amnt_per_status.index)
plt.title('Hello')
fig.autofmt_xdate()

# count of loans per status
loan_count_per_status = data.groupby('loan_status').size()
loan_count_per_status.plot.bar()
plt.bar(range(10), loan_count_per_status)
plt.xticks(range(10), v_shorten(loan_count_per_status.index), rotation=45)

# Beautiful!
fig, ax = plt.subplots(1)
ax.bar(range(10), loan_count_per_status, tick_label=v_shorten(loan_count_per_status.index))
fig.autofmt_xdate()

# Beautiful!
fig, ax = plt.subplots(1)
ax.bar(range(10), np.log1p(loan_count_per_status),
       tick_label=loan_count_per_status.index,
       color = make_binary(loan_count_per_status.index).map({1:'C2', 0:'C0'}))
fig.autofmt_xdate()



# Beautiful!
for status in statuses:
    np.log1p(data[data.loan_status==status].groupby(
            pd.TimeGrouper(freq='M')).size()).plot(label=status)
plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc=3,
           ncol=2, mode="expand")
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#           ncol=2, mode="expand", borderaxespad=0)


# Beautiful!
for status in statuses:
    ls = '-' if status in default else ':'
    np.log1p(data[data.loan_status==status].
             groupby(pd.TimeGrouper(freq='M')).
             size()).plot(label=status,
                 ls=ls, lw=2)
plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc=3,
           ncol=2, mode="expand")


# box plot: status -> loan_amnt





#############################################################################

# Preprocessing


# preprocess predictors
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
# [887379 rows x 1094 columns] with zip_code
# Drop not needed columns
data.drop(drop, axis=1, inplace=True)
# Date columns
date_columns = data[date].apply(pd.to_datetime, format='%b-%Y').astype(np.int64)
date_columns = date_columns.apply(
        lambda column: column.apply(lambda x: -99 if x < 0 else x))
data.drop(date, axis=1, inplace=True)
# Categorical columns
dummy_columns = pd.get_dummies(data[dummy])
data.drop(dummy, axis=1, inplace=True)
# Calculated columns
calculated_columns = data.sub_grade.apply(lambda x: (ord(x[0])-65)*5+int(x[1]))
data.drop('sub_grade', axis=1, inplace=True)

####################################################

# Response variable

####################################################

# 2-class version of y
default = np.array(['Charged Off',
                   'Default',
                   'Does not meet the credit policy. Status:Charged Off',
                   'Late (31-120 days)'])

# data.loan_status.unique() # just checking

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

from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import make_pipeline

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


#####################################################
# skewness

# check the numerical columns
#for feature in data.columns:
#    sns.distplot(data[feature].dropna())
#    sns.plt.show()

from scipy.stats import skew
skewness = pd.Series(0, index=data.columns)
for feature in data.columns:
    skewness[feature] = skew(data[feature], nan_policy='omit')
skewness.sort_values(ascending=False, inplace=True)

for feature in skewness[skewness>0].index #[:33]:
    sns.distplot(np.log(data[feature].dropna()+10))
    sns.plt.show()

for feature in skewness[skewness==0].index #[33:]:
    sns.distplot(data[feature].dropna())
    sns.plt.show()
#sns.distplot(calculated_columns)
#skew(calculated_columns)
take_log = skewness[skewness>0].index
logged_data = data[take_log].apply(lambda x: np.log(x + 10))
data.drop(take_log, axis=1, inplace=True)
X = pd.concat([data, logged_data, date_columns, dummy_columns], axis=1)
######################################################

# Creating final X data frame
X = pd.concat([data, date_columns, dummy_columns], axis=1)
# 157 columns, 887379 rows

# Dealing with missing values
X.fillna(-99, inplace=True)

###########################################################

# Feature Selection

###########################################################
best_features = table.sort_values(by='Importance', ascending=False)
features_selected = best_features.Predictors[:20].values

X_selected = X[features_selected]
X = X_selected

# this is very good! expand on this











# delete [, ], <
# for XGBoost only
import re
features = pd.Series(X.columns)
features = features.apply(lambda x: re.sub('[\[\]<]', '-', x))
X.columns = features


##############################################

# cross validation for binary classification

##############################################

from sklearn.model_selection import train_test_split

# Selecting target y
y = y_binary
# Test-Train split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123)
n_train = y_train.shape[0]
n_test = y_test.shape[0]

from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True, random_state=234)

##############################################

# Reference evaluation measures

##############################################

y_train.value_counts(normalize=True)
#0    0.933705
#1    0.066295
# so predicting 0 should give accuracy of 0.9337

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, np.zeros(y_test.size)) / n_test
# row 0: true solvent
# row 1: true default
# col 0: pred solvent
# col 1: pred default
#array([[ 0.93374128,  0.        ],
#       [ 0.06625872,  0.        ]])
confusion_matrix(y_test, np.zeros(y_test.size))
#array([[248575,      0],
#       [ 17639,      0]])

##############################################

# feature selection

##############################################


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
X_new.shape

from sklearn.feature_selection import RFECV












   

##############################################

# Model fitting

##############################################


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=50,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=345,
                             n_jobs=-1)
'''
grid search: max_features (sqrt, log2, None)
http://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html
'''
# 10 -> 1 min
# 50 -> 2 min
rf.fit(X_train, y_train)

# dump(rf, 'rf_50.pkl')
# rf = load('rf_50.pkl')

# Out-of-fold generalization error
print("%.4f" % rf.oob_score_)
# 50 -> 0.9961
# 50 -> 0.9961 (after logging)

# 50, with 20 top feat: 0.9969

###########################################################

# Model analysis

###########################################################

# 50 trees
# rf.feature_importances_
table = pd.concat([pd.DataFrame(X_train.columns, columns=['Predictors']),
           pd.DataFrame(rf.feature_importances_, columns=['Importance'])],
    axis=1)
table.sort_values(by='Importance', ascending=False)
#                                    Predictors    Importance
#52                                last_pymnt_d  1.555484e-01
#24                     collection_recovery_fee  1.312749e-01
#23                                  recoveries  1.219645e-01
#20                             total_rec_prncp  9.074021e-02
#25                             last_pymnt_amnt  7.197071e-02
#18                                 total_pymnt  5.099405e-02
#19                             total_pymnt_inv  4.273564e-02
#16                                   out_prncp  3.511842e-02
#17                               out_prncp_inv  2.979021e-02
#1                                  funded_amnt  2.902781e-02
#53                                next_pymnt_d  2.411541e-02
#4                                  installment  2.349191e-02
#0                                    loan_amnt  2.107121e-02
#2                              funded_amnt_inv  1.931426e-02
#21                               total_rec_int  1.928359e-02
#51                                     issue_d  1.908051e-02
#50                          last_credit_pull_d  1.546332e-02
#22                          total_rec_late_fee  1.290242e-02


# after logging
#                                    Predictors    Importance
#52                                last_pymnt_d  1.521600e-01
#22                                  recoveries  1.265104e-01
#40                             total_rec_prncp  1.068817e-01
#20                     collection_recovery_fee  1.057137e-01
#30                             last_pymnt_amnt  8.526732e-02
#42                                 total_pymnt  5.018106e-02
#39                             total_pymnt_inv  4.962868e-02
#9                                    out_prncp  3.083747e-02
#10                               out_prncp_inv  2.698445e-02
#0                                    loan_amnt  2.438130e-02
#4                                  installment  2.362969e-02
#53                                next_pymnt_d  2.195636e-02
#2                              funded_amnt_inv  2.176777e-02
#1                                  funded_amnt  2.130587e-02
#38                               total_rec_int  1.924052e-02
#51                                     issue_d  1.738732e-02
#50                          last_credit_pull_d  1.300589e-02
#24                          total_rec_late_fee  1.206558e-02
#3                                     int_rate  8.184655e-03
#18                            total_rev_hi_lim  5.433243e-03
#35                                  open_il_6m  5.298361e-03
#16                                tot_coll_amt  4.601401e-03
#37                                 tot_cur_bal  4.585727e-03
#155                            term_ 36 months  3.564545e-03
#28                                total_bal_il  3.344387e-03
#25                                   revol_bal  3.260373e-03
#17                                         dti  2.972296e-03
#156                            term_ 60 months  2.915756e-03
#19                                  annual_inc  2.732376e-03
#7                                   revol_util  2.725584e-03


###########################################################

# Evaluation

###########################################################

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, recall_score
from sklearn.metrics import f1_score, fbeta_score

# Prediction
'''
y_pred
y_pred_proba
'''
pred = rf.predict(X_test)
pred_proba = rf.predict_proba(X_test)

# Confusion Matrix
# np.around(confusion_matrix(y_test, pred) / n_test, 1)
# 10 trees
#array([[ 93.4,   0. ], -> all solvent caught
#       [  0.4,   6.3]])
confusion_matrix(y_test, pred)
#array([[248556,     19],
#       [   984,  16655]])

# after logging
#array([[248556,     19],
#       [   980,  16659]])

# top 20 features
#array([[248550,     25],
#       [   785,  16854]])

# ROC Curve
'''
ROC
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn-metrics-roc-curve
http://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
http://scikit-learn.org/stable/modules/model_evaluation.html#receiver-operating-characteristic-roc
http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
'''
y_score = pred_proba[:,1]
fpr, tpr, tresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='red', lw=lw,
         label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()



# Precision & Recall curve
'''
See here:
http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py
add area
'''
precision, recall, tresholds = precision_recall_curve(y_test, y_score)
rec = recall_score(y_test, pred)

plt.figure()
lw = 2
plt.plot(precision, recall, color='red', lw=lw, label='Precision-Recall')
plt.plot([0, 1], [1, 0], color='blue', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision-Recall curve')
plt.legend(loc="lower left")
plt.show()

# F1-score
'''
try a range of threshold on your CV set & select the one that maximizes F1 Score
(here it is done on the test set)
'''
f1_score(y_test, pred)
# 0.97076909626089236
for tr in np.linspace(0.1,0.4,20):
    print(tr, f1_score(y_test, y_score>tr))
# best score for 0.31
# Confusion Matrix
confusion_matrix(y_test, y_score>0.31)
#array([[ 93.4,   0. ],
#       [  0.3,   6.3]])
#array([[248515,     60],
#       [   787,  16852]])

# after logging
confusion_matrix(y_test, y_score>0.29)
#array([[248488,     87],
#       [   777,  16862]])

# top 20 features
#array([[248513,     62],
#       [   802,  16837]])


# Fbeta score
# the weighted harmonic mean of precision and recall
fbeta_score(y_test, pred, 1) # = f1_beta
for tr in np.linspace(.01, .1, 20):
    print(tr, fbeta_score(y_test, y_score>tr, 10))
# select: 0.034
#np.around(100*confusion_matrix(y_test, y_score>0.034) / n_test, 1)
#array([[ 89. ,   4.3],
#       [  0.1,   6.5]])
confusion_matrix(y_test, y_score>0.034)
#array([[237015,  11560],
#       [   324,  17315]])
# 324/17315 = 0.0187  

# after logging
for tr in np.linspace(.01, .1, 20):
    print(tr, fbeta_score(y_test, y_score>tr, 10))
confusion_matrix(y_test, y_score>0.038)
#array([[239081,   9494],
#       [   361,  17278]])
    
#####################################################

# Adaboost

from sklearn.ensemble import AdaBoostClassifier

ab = AdaBoostClassifier(random_state=123)
# Fitting
ab.fit(X_train, y_train) # 5 min
# Evaluation
table = pd.concat([pd.DataFrame(X_train.columns, columns=['Predictors']),
           pd.DataFrame(ab.feature_importances_, columns=['Importance'])],
    axis=1)
table.sort_values(by='Importance', ascending=False)
#                     Predictors  Importance
#20              total_rec_prncp        0.22
#4                   installment        0.16
#51                      issue_d        0.16
#52                 last_pymnt_d        0.12
#25              last_pymnt_amnt        0.08
#0                     loan_amnt        0.04
#17                out_prncp_inv        0.04
#3                      int_rate        0.04
#53                 next_pymnt_d        0.04
#21                total_rec_int        0.04
#50           last_credit_pull_d        0.02
#22           total_rec_late_fee        0.02
#23                   recoveries        0.02
# Prediction
pred = ab.predict(X_test)
pred_proba = ab.predict_proba(X_test)
# Confusion Matrix
confusion_matrix(y_test, pred)
#array([[248281,    294],
#       [  1444,  16195]])
# ROC curve
y_score = pred_proba[:,1]
fpr, tpr, tresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='red', lw=lw,
         label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
# Fbeta score
fbeta_score(y_test, pred, 1) # = f1_beta
for tr in np.linspace(.4, .6, 20):
    print(tr, fbeta_score(y_test, y_score>tr, 5))
# select: 0.48
confusion_matrix(y_test, y_score>0.48)
#array([[232943,  15632],
#       [   404,  17235]])


#####################################################

# XGBoost












#####################################################

# LDA
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
#array([[248431,    144],
#       [  6587,  11052]])  # very bad....
    
# after logging
#array([[247933,    642],
#       [  6647,  10992]]) # even worse

# top 20 features
#array([[248434,    141],
#       [  6656,  10983]])

# ROC curve
y_score = pred_proba[:,1]
fpr, tpr, tresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='red', lw=lw,
         label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
# Fbeta score
#fbeta_score(y_test, pred, 1) # = f1_beta
#for tr in np.linspace(.01, .4, 20):
#    print(tr, fbeta_score(y_test, y_score>tr, 5))

#######################################################

# QDA


#####################################################

# SVM(RBF)

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ss = StandardScaler()
std_X_train = ss.fit_transform(X_train)

#### TO RUN IT AT ALL YOU NEED A VERY SIMPLE VERSION

svm = SVC(random_state=123) # grid search parameters
# Fitting
svm.fit(X_train[:10000], y_train[:10000])
# 10,000 -> 2 min, too much!
# Evaluation
# Prediction
std_X_test = ss.transform(X_test)
pred = svm.predict(std_X_test) ####################### DONT RUN IT
dec_func = svm.decision_function(std_X_test)
# Confusion Matrix
confusion_matrix(y_test, pred)

# ROC curve
y_score = dec_func[:,1]
fpr, tpr, tresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='red', lw=lw,
         label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
# Fbeta score
fbeta_score(y_test, pred, 1) # = f1_beta
for tr in np.linspace(.4, .6, 20):
    print(tr, fbeta_score(y_test, y_score>tr, 5))
# select: 0.48
confusion_matrix(y_test, y_score>0.48)
#array([[232943,  15632],
#       [   404,  17235]])


#####################################################

# KNN
    

    


















    
#################################################

# Multiclass modeling

#################################################
'''
Multiclass
http://scikit-learn.org/stable/modules/multiclass.html
http://scikit-learn.org/stable/auto_examples/plot_multilabel.html#sphx-glr-auto-examples-plot-multilabel-py
http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
'''


###################################################################
# 10-class version integer
#tmp_data = pd.read_csv('../input/loan.csv')
#tmp_le = LabelEncoder()
#tmp_y_multi = tmp_le.fit_transform(tmp_data.loan_status)
#tmp_y_multi
#
## class weights
#tmp = response_pipe.transform(default)
#tmp
#tmp.shape
#tmp2 = tmp.sum(axis=0)
#tmp2
#tmp3 = 10*tmp2
#tmp3
##n_classes = tmp.shape[1]
##classes = np.arange(n_classes)
##list(zip(classes, tmp3))
##class_weights = dict(zip(classes, tmp3))
##class_weights
#tmp2 = tmp_le.transform(default)
#tmp3 = 9*np.bincount(tmp)+1
#tmp4 = dict(zip(np.arange(tmp3.shape[0]),tmp3))
#
#y = tmp_y_multi

####################################################################

##############################################

# cross validation for multiclass classification

##############################################

from sklearn.model_selection import train_test_split

# Selecting target y
y = y_multi
# Test-Train split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123)
n_train = y_train.shape[0]
n_test = y_test.shape[0] # for sparse matrix use shape[0]

# One-Vs-The-Rest

from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
#from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

## don't run this
#clf1 = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train[:10000], y_train[:10000])
## 10,000 samples -> 1 min
#pred = clf1.predict(X_test)
#pred_binary = make_binary(response_pipe.inverse_transform(pred))
#resp_binary = make_binary(response_pipe.inverse_transform(y_test))
## Confusion Matrix
#np.around(100*confusion_matrix(resp_binary, pred_binary) / n_test, 1)



## I tried normalization but it was a disaster
#from sklearn.preprocessing import Normalizer
#std_X_test = norm.transform(X_test)
#norm = Normalizer()
#std_X_train = norm.fit_transform(X_train)
##array([[248575,      0],
##       [ 17476,    163]])


ss = StandardScaler()
std_X_train = ss.fit_transform(X_train)
clf2 = OneVsRestClassifier(
        SGDClassifier(loss='log', random_state=0,
                      #class_weight=tmp4,
                      n_jobs=-1)).fit(std_X_train, y_train)
# 1 min
clf2.coef_.shape
clf2.intercept_
std_X_test = ss.transform(X_test)
pred = clf2.predict(std_X_test)
pred_binary = make_binary(response_pipe.inverse_transform(pred))
resp_binary = make_binary(response_pipe.inverse_transform(y_test))
# Confusion Matrix
np.around(100*confusion_matrix(resp_binary, pred_binary) / n_test, 1)
# no weights
#array([[ 92.6,   0.8],
#       [  1.4,   5.2]])
confusion_matrix(resp_binary, pred_binary)
#array([[246406,   2169],
#       [  3866,  13773]])

# after logging
#array([[246466,   2109],
#       [  3944,  13695]])

###############################################################################

# XGBoost

# idea: transfom PCA
# idea: select most important features
# idea: check most important features, take log of them if needed
# idea: check for correlated columns -> vif?
# try different strategies for missing values
# try checking for and removing outlayers
# try checking for skew variables to correct

# grid search

# try optional score or class weights (so far failure)

# ensembling










# SVM + StandardScaler (normalization)

# SGDClassifier
# Linear classifiers (SVM, logistic regression, a.o.) with SGD training
# data should have zero mean and unit variance.

# idea: use class_weight to concentrate on defaults
# e.g. http://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html#sphx-glr-auto-examples-svm-plot-separating-hyperplane-unbalanced-py
# idea: run a classifer with F10_score goal, e.g. adaboost
# idea: once selecting best classifier run 10x CV on whole data to get CV score

# clf: adaboost, xgboost, random forest, LDA, QDA
# clf with standardization: SVM, KNN


    

# next: check all what people done on this -> look for ideas
# next: grid search CV for randomforest (simple, just needs time)
# next: mode models: xgboost, logistic regression, SVM(rfb)
# next: ensembling
# next: multilabel classification -> maybe more accurate?





# fit a model (classification std models + xgboost)
# evaluate using cv
# (read again about inner and outer cv)
# check: do you need to standardize the variable?


# Next: check for correlated columns -> vif?
# try PCA
# try selecting some columns
# try different strategies for missing values
# try checking for and removing outlayers
# try checking for skew variables to correct
# try standardizing the variables

# Next: visualisations/info extraction
# select most important variables or maybe two top PCA?
# make some visualisations


