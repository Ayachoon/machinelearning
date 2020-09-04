#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 11:07:53 2020

@author: ayakanakatsuka
"""
#Importing modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Downloading data
url = 'https://datahack-prod.s3.amazonaws.com/train_file/train_ctrUa4K.csv'
train = pd.read_csv(url)

url1='https://datahack-prod.s3.amazonaws.com/test_file/test_lAUu6dG.csv'
test=pd.read_csv(url1)

#Understanding the sample universe
train['Gender'].value_counts(normalize=True).plot.bar( title= 'Gender') 
train['Education'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Education') 
train['Self_Employed'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Self Employed') 
train['Credit_History'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Credit history') 

#Analyze the data
pd.crosstab(train['Loan_Status'],train['Credit_History']).apply(lambda r: r/r.sum())
pd.crosstab(train['Loan_Status'],train['Female']).apply(lambda r: r/r.sum())
pd.crosstab(train['Loan_Status'],train['Property_Area']).apply(lambda r: r/r.sum())
pd.crosstab(train['Loan_Status'],train['Education']).apply(lambda r: r/r.sum())

train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()
sns.boxplot(x="Loan_Status", y="ApplicantIncome", data=train)
sns.boxplot(x="Loan_Status", y="CoapplicantIncome", data=train)
sns.boxplot(x="Loan_Status", y="LoanAmount", data=train)
train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()

#Cleanse the data
train.isna().sum()
train['Loan_Status'].replace("Y",0,inplace=True)
train['Loan_Status'].replace("N",1,inplace=True)
train=train.drop(['Loan_ID'],axis=1)
test=test.drop(['Loan_ID'],axis=1)

median = train['LoanAmount'].median()
train['LoanAmount'].fillna(median, inplace=True)
median1 = test['LoanAmount'].median()
test['LoanAmount'].fillna(median1, inplace=True)

train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True) 
train['Married'].fillna(train['Married'].mode()[0], inplace=True) 
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True) 
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True) 

test['Self_Employed'].fillna(test['Self_Employed'].mode()[0], inplace=True) 
test['Gender'].fillna(test['Gender'].mode()[0], inplace=True) 
test['Married'].fillna(test['Married'].mode()[0], inplace=True) 
test['Dependents'].fillna(test['Dependents'].mode()[0], inplace=True) 
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0], inplace=True) 
test['Credit_History'].fillna(test['Credit_History'].mode()[0], inplace=True) 

#Building a model 
train=pd.get_dummies(train) 
test=pd.get_dummies(test)

X = train.drop('Loan_Status',1) 
y = train.Loan_Status

from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression() 
model.fit(x_train, y_train)
LogisticRegression(C=1.0)
pred_test = model.predict(test)

#Check Accuracy

pred_cv = model.predict(x_cv)


