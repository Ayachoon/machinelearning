#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 14:32:54 2020

@author: ayakanakatsuka
"""

import pandas as pd

test=pd.read_csv('/Users/ayakanakatsuka/Documents/datascience/houseprice/test.csv')

train=pd.read_csv('/Users/ayakanakatsuka/Documents/datascience/houseprice/train.csv')

import matplotlib.pyplot as plt

import numpy as np

#Analyzing Data
train.groupby("SaleCondition")["SalePrice"].mean()
train.SalePrice.describe()

plt.hist(train["SalePrice"])
 
plt.hist(train.SalePrice)

target=np.log(train["SalePrice"])

print(target.skew)

#Find what is postiviely correalted and negatively
#1 select the data that are in the number
numeric_features=train.select_dtypes(include=[np.number])
#check what type of data they are 
numeric_features.dtypes
corr = numeric_features.corr()
SalePriceCorr=corr["SalePrice"].sort_values(ascending=False)

#Plot the relationship between living area vs sale price
plt.scatter(x=train["GrLivArea"],y=train["SalePrice"])
plt.xlabel('Living Area')
plt.ylabel('Sale Price')

#use pivot table to understand the meidan price vs quality 
quality_pivot = train.pivot_table(index='OverallQual',values='SalePrice', aggfunc=np.median)

#Check Nullvalues in the dataframe
Nullvalues=train.isnull().sum()
Nullvalues=train.isnull().sum().sort_values(ascending=False)[:25]


#Chcek the breakdown of values in a specific data
train["Street"].value_counts()

#change values to machine readable 
categoricals = train.select_dtypes(exclude=[np.number])
categoricals.describe()
def encode(x):
 return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)

train['enc_street'] = pd.get_dummies(train["Street"], drop_first=True)
test['enc_street'] = pd.get_dummies(train["Street"], drop_first=True)

#interpolate the value that are missing 
data = train.select_dtypes(include=[np.number]).interpolate().dropna()

#Model building 
y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                          X, y, random_state=42, test_size=.33)
 
from sklearn import linear_model
lr = linear_model.LinearRegression()

model = lr.fit(X_train, y_train)

print ("R^2 is: \n", model.score(X_test, y_test))

#RMSE
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))

actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.7,
            color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()



