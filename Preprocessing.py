# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:41:43 2019

@author: Amirhossein Forouzani
"""
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import sklearn
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import random
import math
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
from IPython.core.interactiveshell import InteractiveShell
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
import seaborn as sns
#--------------------------------------------------
# costum encoder
def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            result[column].str.rstrip()
            result[column].str.lstrip()
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders

#Costum Imputer    
class ImputeCategorical(BaseEstimator, TransformerMixin):
    """
    Encodes a specified list of columns or all columns if None.
    """
    def __init__(self, columns=None):
        self.columns = columns
        self.imputer = None
    def fit(self, data, target=None):
        """
        Expects a data frame with named columns to impute.
        """
        # Encode all columns if columns is None
        if self.columns is None:
            self.columns = data.columns
        # Fit an imputer for each column in the data frame
        self.imputer = Imputer(missing_values=0, strategy='most_frequent')
        self.imputer.fit(data[self.columns])
        return self
    def transform(self, data):
        """
        Uses the encoders to transform a data frame.
        """
        output = data.copy()
        output[self.columns] = self.imputer.transform(output[self.columns])
        return output
#--------------------------------------------------
#File Importer/Imputer/Encoder
#inputs: File Name
#outputs:         
def import_file (data_name):
    dataset_name = data_name
    df_train = pd.read_csv(dataset_name + ".train_SMALLER.csv")
    df_test = pd.read_csv(dataset_name + ".test_SMALLER.csv")
    df_train.columns = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "label"]
    df_test.columns = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "label"]
    encoded_train, _ = number_encode_features(df_train)
    imputer = ImputeCategorical(['Workclass', 'Country', 'Occupation'])
    encoded_train = imputer.fit_transform(encoded_train)
    
    encoded_test, _ = number_encode_features(df_test)
    imputer = ImputeCategorical(['Workclass', 'Country', 'Occupation'])
    encoded_test = imputer.fit_transform(encoded_test)
   
    y_train = encoded_train['label']
    x_train = encoded_train.drop(['label'], axis = 1)
    y_test = encoded_test['label']
    x_test = encoded_test.drop(['label'], axis = 1)
    return x_train, y_train, x_test, y_test, encoded_train, encoded_test
#scaler = StandardScaler().fit(x_train)
def distribution_finder (data_name):
    og_data = pd.read_csv(data_name + ".train_SMALLER.csv")
    og_data.columns = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "label"]
    fig = plt.figure(figsize=(20,15))
    cols = 5
    rows = math.ceil(float(og_data.shape[1]) / cols)
    for i, column in enumerate(og_data.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if  og_data.dtypes[column] == np.object:
            og_data[column].value_counts().plot(kind="bar", axes=ax)
        else:
            og_data[column].hist(axes=ax)
            plt.xticks(rotation="vertical")
        plt.subplots_adjust(hspace=0.7, wspace=0.2)
def frequency_finder (data_name,frame_name):
    og_data = pd.read_csv(data_name + ".train_SMALLER.csv")
    og_data.columns = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "label"]
    og_data.head()
    f, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True)  
    sns.countplot(y = frame_name, hue='label', data=og_data,)
def corellation_ploter(data):
    a,b,c,d,e,f = import_file(data)
    sns.heatmap(e.corr(), square=True)
    plt.show()
   
x_train, y_train, x_test, y_test, encoded_train, encoded_test = import_file("adult")


#distribution_finder("adult")
#frequency_finder("adult", "Occupation")




'''
#this is the pipeliner
# we need to encode our target data as well.
yencode = LabelEncoder().fit(dataset.label)
# construct the pipeline
census = Pipeline([
        ('encoder',  EncodeCategorical(dataset.categorical_features.keys())),
        ('imputer', ImputeCategorical(['workclass', 'native-country', 'occupation'])),
        ('classifier', LogisticRegression())
    ])
# fit the pipeline
census.fit(dataset.data, yencode.transform(dataset.label))
'''