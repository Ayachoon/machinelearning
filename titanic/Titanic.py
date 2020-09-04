# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#download data
import pandas as pd
import numpy as np
test=pd.read_csv('/Users/ayakanakatsuka/Documents/datascience/titanic/test.csv')

gender=pd.read_csv('/Users/ayakanakatsuka/Documents/datascience/titanic/gender_submission.csv')

train=pd.read_csv('/Users/ayakanakatsuka/Documents/datascience/titanic/train.csv')

#Analyzing Data
women= train.loc[train.Sex=="female"]["Survived"]
rate_women= sum(women) / len(women)
male= train.loc[train.Sex=="male"]["Survived"]
rate_male= sum(male) / len(male)
class_stats=train.groupby("Pclass").sum()
class_stats["Percentage"] = class_stats["Survived"]  / class_stats["Survived"].count()

#Cleaning Data
train_updated=train.loc[:,["Survived","Pclass","Sex","Age","SibSp","Parch"]]

train_updated['Sex'].replace("male",0,inplace=True)
train_updated['Sex'].replace("female",1,inplace=True)

train_updated=train_updated.dropna()

train_updated["Adult"] = 0

train_updated["Adult"][train_updated["Age"] >= 18] = 1

train_updated = train_updated.drop('Age', axis=1,)

#Find correlation
numeric_features=train.select_dtypes(include=[np.number])
numeric_features.dtypes
corr = numeric_features.corr()
SurvivedCorr=corr["Survived"].sort_values(ascending=False)

#Clean the test data
test_updated=test.loc[:,["Pclass","Sex","Age","SibSp","Parch"]]
test_updated['Sex'].replace("male",0,inplace=True)
test_updated['Sex'].replace("female",1,inplace=True)
test_updated=test_updated.dropna()
test_updated["Adult"] = 0
test_updated["Adult"][test_updated["Age"] >= 18] = 1
test_updated = test_updated.drop('Age', axis=1,)

#Machine Learning
from sklearn.ensemble import RandomForestClassifier

y = train["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

model.predict(X_test.iloc[[1]])


model.predict(X_test.iloc[[55]])

#Another model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
 
X_train=train_updated.loc[:,["Pclass","Sex","Adult","SibSp","Parch"]]
y_train=train_updated["Survived"]
X_test=test_updated

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
knn.score(X_test,y_test)