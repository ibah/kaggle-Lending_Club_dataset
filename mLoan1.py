# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:33:55 2017
@author: msiwek
Kaggle dataset - Lending Club Loan Data
https://www.kaggle.com/wendykan/lending-club-loan-data


"""

import os
os.chdir('/home/michal/Dropbox/cooperation/_python/LendingClub-dataset/models')
os.chdir('D:\\data\\Dropbox\\cooperation\\_python\\LendingClub-dataset\\Models')
os.chdir('G:\\Dropbox\\cooperation\\_python\\LendingClub-dataset\\Models')

import numpy as np
import pandas as pd
import seaborn as sns

# Loading data
data = pd.read_csv('../input/loan.csv')
# Columns (19,55) have mixed types. Specify dtype option on import or set low_memory=False.
# interactivity=interactivity, compiler=compiler, result=result)
data.info()
data.head()
n = data.shape[0]
# looks like a lot of Nans
data.dtypes.value_counts()
# Exploration

# missing values

null_col = data.isnull().sum().sort_values(ascending=False)
null_col/n
'''
Comments on missing values
- many variables have over 97% of nan, then 85, 84, 75, 51, 28, and 3x 8, 6, 2, and below 1%

'''
(data.emp_length=='n/a').sum()/n # 5% missing values
for col in data:
    if data[col].dtype == 'object':
        q = (data[col] == 'n/a').sum()/n
        if q > 0:
            print(col, q)
#emp_title # <1%
#emp_length # 5%
#desc # <1%
#title # <1%
# replace with nan

# object columns

data.loc[:,data.dtypes == 'object']
tmp = data.select_dtypes(['object'])
tmp.shape # 23 columns
tmp.info()
len(tmp.term.unique())
pd.unique(tmp)
tmp.apply(pd.unique)
## 1
#obj_col = pd.DataFrame({'unique': 0}, index=tmp.columns)
#for col in tmp:
#    obj_col.loc[col] = len(tmp[col].unique())
# 2
obj_col = pd.Series(0, index=tmp.columns)
for col in tmp:
    obj_col[col] = len(tmp[col].unique())
obj_col.sort_values(ascending=False, inplace=True)
obj_col/n
tmp.loc[:,obj_col.index[:5]].head()
# reviewing object column
tmp.emp_title
tmp.desc
tmp.title
tmp.zip_code.unique().size # 935 values
tmp.zip_code.apply(len).unique() # all 5 characters long
import re
x = tmp.zip_code[0]
re.search('..$',x).group()
tmp2 = tmp.zip_code.apply(lambda x: re.search('..$',x).group())
tmp2.unique() # all containg 'xx' at the end
'''
Comments on the object columns
url - drop it totally, 100% unique
emp_title - 1/3 unique values, job title, leave it but maybe it could be analyzed
  - some language tool: lower case letters -> extract words
  - try just length of this field (on some competitions it worked)
  34% emp_title
  6% missing
  initial model: drop
  look - the same info (or similar/better) is provided by purpose
desc - Loan description provided by the borrower
  14% unique
  86% missing
  maybe you can extract some keywords
  initial model: drop
title - The loan title provided by the borrower
  7% unique, <1% missing
  extract keywords
  consolidation, repayment, medical, card, car, home/house, 
  initial model: leave it -> no, drop it, use purpose
  next model: extract keywords
zip_code
  935 unique values
  can be useful
'''
def plot_time(x):
    pd.to_datetime(x, format='%b-%Y').value_counts().plot()
tmp.earliest_cr_line
plot_time(tmp.earliest_cr_line) # interesting; maybe bucket it and create dummies
pd.to_datetime(tmp.earliest_cr_line[0])
pd.to_datetime(tmp.earliest_cr_line, format='%b-%Y') # converted to timestamp
tmp.last_credit_pull_d
plot_time(tmp.last_credit_pull_d) # very uninteresting
tmp.last_credit_pull_d.value_counts()
tmp.issue_d # The month which the loan was funded
q = pd.to_datetime(tmp.issue_d, format='%b-%Y').value_counts()
q
q.plot()
sns.tsplot(q.values, q.index)
tmp.next_pymnt_d # 28% missing
plot_time(tmp.next_pymnt_d)
tmp.last_pymnt_d # 2% missing
plot_time(tmp.last_pymnt_d)
'''
earliest_cr_line - The month the borrower's earliest reported credit line was opened
  extract as a date
  create: difference between this and current credit
  convert to timestamp
last_credit_pull_d - The most recent month LC pulled credit for this loan
  convert to timestamp
... to timestamp -> you may need to convert to np.int
'''
tmp.addr_state # nice, just state 2-letter codes
tmp.addr_state.apply(len).unique() # ok
tmp.sub_grade
# you have to order that from A1 to G5 - this is order variable
x = tmp.sub_grade[0]
x
x[0]
int(x[1])
(ord(x[0])-65)*5+int(x[1])
tmp.sub_grade.apply(lambda x: (ord(x[0])-65)*5+int(x[1])) # numerical value
tmp.purpose # nice
tmp.purpose.value_counts() # nicely collected values
tmp.purpose.isnull().sum() # no null
tmp.emp_length
# Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years. 
# this can be ordinal... or just dummies
tmp.emp_length.value_counts()
# see n/a -> change to np.Nan
(tmp.emp_length=='n/a').sum()/n # 5% missing values
tmp.loan_status # response variable - see below
tmp.grade # LC assigned loan grade
# ordinal
tmp.grade.value_counts()
np.all(tmp.grade == tmp.sub_grade.apply(lambda x: x[0])) # grade is correct
tmp.home_ownership.value_counts() # nice dummies
tmp.verification_status_joint.value_counts()
# verified_status_joint - Indicates if the co-borrowers' joint income was verified by LC, not verified, or if the income source was verified
# merge source verified?
# 99% NaN
tmp.verification_status.value_counts() # nice
tmp.pymnt_plan.value_counts()
# Indicates if a payment plan has been put in place for the loan
# very low variation, may be useless
tmp.application_type.value_counts()
tmp.initial_list_status.value_counts() # The initial listing status of the loan. Possible values are – W, F
tmp.term.value_counts()

# check columns 19 and 55

data.columns[19] # desc
data.columns[55] # verification_status_joint
data.iloc[:,55] # a lot of nans
data.iloc[:,55].isnull().sum()/n # 99%

# numerical columns

for col in data:
    if data[col].dtype != 'object':
        print(col)
num_col_list = [col for col in data if data[col].dtype != 'object']
num_col = pd.Series(0, index=num_col_list)
for col in num_col_list:
    num_col[col] = data[col].unique().size
num_col.sort_values(ascending=False, inplace=True)
num_col/n
'''
comments on numerical columns
drop id, member_id
policy_code: 0/1
'''

         
# Response variable

data.loan_status.unique()
data['loan_status'].unique()[:,None]
pd.DataFrame(data['loan_status'].unique()).values
data.loan_status.value_counts()
'''
y
> Default
Charged off
Default
Does not meet the credit policy. Status:Charged Off
Late (31-120 days)
> Solvent
Fully Paid
Current
In Grace Period
Late (16-30 days)
Does not meet the credit policy. Status:Charged Off
Issued
'''
default = np.array(['Charged off',
                   'Default',
                   'Does not meet the credit policy. Status:Charged Off',
                   'Late (31-120 days)'])
def is_default(x):
    return x in default
ax = sns.countplot(data.loan_status)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30) # just counts

#### demanding frequencies and counts ###
# Make twin axis
ncount = data.shape[0]
ax2=ax.twinx()
# Switch so count axis is on right, frequency on left
ax2.yaxis.tick_left()
ax.yaxis.tick_right()
# Also switch the labels over
ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')
ax2.set_ylabel('Frequency [%]')
for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text

# Use a LinearLocator to ensure the correct number of ticks
import matplotlib.ticker as ticker
ax.yaxis.set_major_locator(ticker.LinearLocator(11))
# Fix the frequency range to 0-100
ax2.set_ylim(0,100)
ax.set_ylim(0,ncount)
# And use a MultipleLocator to ensure a tick spacing of 10
ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))
# Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars
ax2.grid(None)
### ###

tmp = data.loan_status.value_counts()
tmp
tmp.groupby(tmp.index)
data.loan_status.value_counts().groupby(tmp.index).count()
data.loan_status.value_counts().groupby(is_default).count()
data.loan_status[default].value_counts()
is_default(data.loan_status)

#############################################

# preprocessing

#############################################

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline

# load data
data = pd.read_csv('../input/loan.csv')
# recognized missing data
data.replace('n/a', np.nan, inplace=True)
# preprocess predictors
drop = ['url','emp_title','desc','title','id','member_id','zip_code']
# title is represented by purpose
# zip_code requires to much memory, represented by addr_state (state)
# possibly you could aggregate zip_codes that perform well/bad in the training set
# and create an additional variables
dummy = ['addr_state', 'purpose', 'emp_length', 'grade',
         'home_ownership', 'verification_status_joint',
         'verification_status', 'pymnt_plan', 'application_type',
         'initial_list_status', 'term']
# [887379 rows x 1094 columns] with zip_code
date = ['earliest_cr_line', 'last_credit_pull_d', 'issue_d',
        'last_pymnt_d', 'next_pymnt_d']
data.drop(drop, axis=1, inplace=True)
# data.drop('zip_code', axis=1, inplace=True)
# pd.to_datetime(data.next_pymnt_d[:10]) # missing values are NaT
# data.ix[0:10,date]
date_columns = data[date].apply(pd.to_datetime, format='%b-%Y').astype(np.int64)
date_columns = date_columns.apply(lambda column: column.apply(lambda x: -99 if x < 0 else x))
#tmp = date_columns.astype(np.int64)
#date_columns = tmp.apply(lambda column: column.apply(lambda x: -99 if x < 0 else x))
#tmp[:10]
#tmp[:10].apply(lambda x: -99 if x < 0 else x)
#for col in tmp:
#    print(tmp.ix[:10,col].apply(lambda x: -99 if x < 0 else x))
    
data.drop(date, axis=1, inplace=True)
# pd.get_dummies(data.ix[:10,dummy])
dummy_columns = pd.get_dummies(data[dummy])
data.drop(dummy, axis=1, inplace=True)
calculated_columns = data.sub_grade.apply(lambda x: (ord(x[0])-65)*5+int(x[1]))
data.drop('sub_grade', axis=1, inplace=True)

# response variable
default = np.array(['Charged off',
                   'Default',
                   'Does not meet the credit policy. Status:Charged Off',
                   'Late (31-120 days)'])
y_simple = data.loan_status.apply(lambda x: 1 if x in default else 0)

#y_pipe = make_pipeline(LabelEncoder(), OneHotEncoder())
#y_pipe.fit(data.loan_status)
le = LabelEncoder()
#le.fit(data.loan_status)
#le.transform(data.loan_status[:10])
tmp = le.fit_transform(data.loan_status).reshape(-1,1)
enc = OneHotEncoder()
y_full = enc.fit_transform(tmp)
# print(y_full[:10,:10])
# you can use also: label_binarizer(tmp)

data.drop('loan_status', axis=1, inplace=True)

# creating final X data frame
X = pd.concat([data, date_columns, dummy_columns], axis=1)
# NaT -> -9223372036854775808
# X.info()
# 157 columns, 887379 rows
# X.ix[:10,X.dtypes=='object']

# deal with NaNs
X.min()
X.isnull().sum()
X.fillna(-99, inplace=True)
data.min() # min value is -4
# use -99 for NaN and NaT
# date_columns.isnull().sum()
y = y_simple


##############################################

# cross validation

##############################################

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123)
n_train = len(y_train)
n_test = len(y_test)

from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True, random_state=234)

##############################################

# reference evaluation measures

##############################################

y_train.value_counts(normalize=True)
# 0    0.984716
# 1    0.015284
# so predicting 0 should give accuracy of 0.9847

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, np.zeros(y_test.size)) / n_test
# row 0: true solvent
# row 1: true default
# col 0: pred solvent
# col 1: pred default

##############################################

# model fitting

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
# 10 -> 1 min
# 50 -> 2 min
rf.fit(X_train, y_train)
print("%.4f" % rf.oob_score_)
# 10 -> 0.9954
# 50 -> 0.9969
pred = rf.predict(X_test)
np.around(100*confusion_matrix(y_test, pred) / n_test, 1)
# 10 trees
#array([[ 98.5,   0. ],
#       [  0.3,   1.2]]) -> all solvent caught, 1.2 vs 0.3 default caught
# 50 trees
rf.feature_importances_
type(X_train.columns)
table = pd.concat([pd.DataFrame(X_train.columns, columns=['Predictors']),
           pd.DataFrame(rf.feature_importances_, columns=['Importance'])],
    axis=1)
table.sort_values(by='Importance', ascending=False)
#                                    Predictors  Importance
#52                                last_pymnt_d    0.427079
#53                                next_pymnt_d    0.086727
#16                                   out_prncp    0.061864
#17                               out_prncp_inv    0.048435
#51                                     issue_d    0.030434
#25                             last_pymnt_amnt    0.029226
#20                             total_rec_prncp    0.024663
#19                             total_pymnt_inv    0.024312
#21                               total_rec_int    0.024295
#18                                 total_pymnt    0.023778
#22                          total_rec_late_fee    0.019128


###########################################################

# Evaluation

###########################################################

# ROC Curve

from sklearn.metrics import roc_curve

pred_proba = rf.predict_proba(X_test)
rc = roc_curve(y_test, )

    
    
# next
# Calculate performance measurement(s), e.g. ROC, AUC + sensitivity, specificity
# +plots

# next: check all what people done on this -> look for ideas
# next: grid search CV for randomforest (simple, just needs time)
# next: mode models: xgboost, logistic regression, SVM(rfb)
# next: ensembling
# next: multilabel classification -> maybe more accurate?





# fit a model (classification std models + xgboost)
# evaluate using cv
# (read again about inner and outer cv)
# check: do you need to standardize the variable?


# Next: add ROC curve, AUC (from scikit)

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


