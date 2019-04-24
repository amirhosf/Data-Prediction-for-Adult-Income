# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:03:16 2019

@author: Amirhossein Forouzani
"""
import sklearn.metrics as metrics
import sklearn as skl
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
from sklearn import linear_model
from IPython.core.interactiveshell import InteractiveShell
import sklearn.preprocessing as preprocessing
from sklearn.pipeline import Pipeline
#-------------------------------------------------------------------------
#the import function would encode and impute the data attributes
x_train, y_train, x_test, y_test, encoded_train, encoded_test ,encoders_train,encoders_test= import_file("adult")

#distribution_finder("adult")
#frequency_finder("adult", "Occupation")
#impute and encode

#Scaling the data
'''
scalar =preprocessing.StandardScaler()
x_train  = scalar.fit_transform(x_train)
x_test = scalar.fit_transform(x_test)
'''
# using dummy variables to encode the data
binary_data_train = pd.get_dummies(x_train)
binary_data2_test = pd.get_dummies(x_test)


scalar =preprocessing.StandardScaler()
x_train  = pd.DataFrame(scalar.fit_transform(x_train))

x_test = pd.DataFrame(scalar.fit_transform(x_test))

# running the linear regression model on the data set
cls = linear_model.LogisticRegression()

cls.fit(x_train, y_train)
y_pred = cls.predict(x_test)
cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders_train["label"].classes_, yticklabels=encoders_train["label"].classes_)
plt.ylabel("Real value")
plt.xlabel("Predicted value")
print ("F1 score: %f" % skl.metrics.f1_score(y_test, y_pred))
trainerror = 1-accuracy_score ( y_test ,y_pred )
print ("train error is: ",trainerror )
coefs = pd.Series(cls.coef_[0], index=encoded_train.drop(['label'],axis = 1).columns)
coefs.sort_values()
ax = plt.subplot(2,1,2)
coefs.plot(kind="bar")
plt.show()
#
#


