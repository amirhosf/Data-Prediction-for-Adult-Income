#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:22:55 2019

@author: amiru
"""

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
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import KernelPCA
from sklearn.neural_network import MLPClassifier
#from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
#-------------------------------------------------------------------------
#the import function would encode and impute the data attributes
x_train, y_train, x_test, y_test, encoded_train, encoded_test ,encoders_train,encoders_test= import_file("adult")
og_x_train = x_train
og_x_test = x_test 
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


scalar =preprocessing.StandardScaler(with_std=True)
x_train  = pd.DataFrame(scalar.fit_transform(x_train))
x_test = pd.DataFrame(scalar.fit_transform(x_test))
#fdeature dimesnasio reduction for future use

# running the linear regression model on the data set(F1 score: 0.536377)

cls = linear_model.LogisticRegression()

cls.fit(x_train, y_train)
y_pred = cls.predict(x_test)
cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders_train["label"].classes_, yticklabels=encoders_train["label"].classes_)
plt.ylabel("Real value")
plt.xlabel("Predicted value")
print ("F1 score for Linear Regression: %f" % skl.metrics.f1_score(y_test, y_pred))
trainerror = accuracy_score ( y_test ,y_pred )
print ("Accuracy is: ",trainerror )
coefs = pd.Series(cls.coef_[0], index=encoded_train.drop(['label'],axis = 1).columns)
coefs.sort_values()
ax = plt.subplot(2,1,2)
coefs.plot(kind="bar")
plt.show()
#
#Now we can trey to classify using perceptron
cls = OneVsRestClassifier(Perceptron(tol=1e-3, random_state=0))

cls.fit(x_train, y_train)
y_pred = cls.predict(x_test)
cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders_train["label"].classes_, yticklabels=encoders_train["label"].classes_)
plt.ylabel("Real value")
plt.xlabel("Predicted value")
print ("F1 score for Perceptron using One vurses rest classifier: %f" % skl.metrics.f1_score(y_test, y_pred))
trainerror = accuracy_score ( y_test ,y_pred )
print ("train error is: ",trainerror )
coefs = pd.Series(cls.coef_[0], index=encoded_train.drop(['label'],axis = 1).columns)
coefs.sort_values()
ax = plt.subplot(2,1,2)
coefs.plot(kind="bar")
plt.show()
#now we can run the MSE_binary Using One Vs. Rest Classifier
binary_model = MSE_binary ( )
mc_model = OneVsRestClassifier (binary_model)
mc_model.fit(x_train, y_train)
y_pred = mc_model.predict(x_test)
cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders_train["label"].classes_, yticklabels=encoders_train["label"].classes_)
plt.ylabel("Real value")
plt.xlabel("Predicted value")
print ("F1 score for MSE_binary with linear Regression is : %f" % skl.metrics.f1_score(y_test, y_pred))
trainerror = accuracy_score ( y_test ,y_pred )
print ("Accuracy is: ",trainerror )
coefs = pd.Series(cls.coef_[0], index=encoded_train.drop(['label'],axis = 1).columns)
coefs.sort_values()
ax = plt.subplot(2,1,2)
coefs.plot(kind="bar")
plt.show()



# Now we can classify iusing support vector machines(F1 Score: 0.228)
cls = SVC(kernel ='rbf', C = 50, gamma = 5)
cls.fit(x_train, y_train)
y_pred = cls.predict(x_test)
cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders_train["label"].classes_, yticklabels=encoders_train["label"].classes_)
plt.ylabel("Real value")
plt.xlabel("Predicted value")
print ("F1 score for SVM with RBF Kernel: %f" % skl.metrics.f1_score(y_test, y_pred))
trainerror = accuracy_score ( y_test ,y_pred )
print ("Accuracy is: ",trainerror )
#coefs = pd.Series(cls.coef_[0], index=encoded_train.drop(['label'],axis = 1).columns)
#coefs.sort_values()
#ax = plt.subplot(2,1,2)
#coefs.plot(kind="bar")






#SVM with Sigmoid Kernel(F1 Score: 36%)
cls = SVC(kernel ='sigmoid', C = 50, gamma = 5)
cls.fit(x_train, y_train)
y_pred = cls.predict(x_test)
cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders_train["label"].classes_, yticklabels=encoders_train["label"].classes_)
plt.ylabel("Real value")
plt.xlabel("Predicted value")
print ("F1 score for SVM with sigmoid Kernel: %f" % skl.metrics.f1_score(y_test, y_pred))
trainerror = accuracy_score ( y_test ,y_pred )
print ("Accuracy is: ",trainerror )
#coefs = pd.Series(cls.coef_[0], index=encoded_train.drop(['label'],axis = 1).columns)
#coefs.sort_values()
#ax = plt.subplot(2,1,2)
#coefs.plot(kind="bar")
plt.show()
#SVM with Sigmnoid Kernel with Feature reduction(f1 score:39%)
'''
transformer = KernelPCA(n_components=7, kernel='sigmoid')
X_transformed = transformer.fit_transform(x_train)
transformer2 = KernelPCA(n_components=7, kernel='sigmoid')
X_transformed_t = transformer2.fit_transform(x_test)
'''



cls = SVC(kernel ='sigmoid', C = 50, gamma = 5)
cls.fit(xtrain_new, y_train)
y_pred = cls.predict(xtest_new)
cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders_train["label"].classes_, yticklabels=encoders_train["label"].classes_)
plt.ylabel("Real value")
plt.xlabel("Predicted value")
print ("SVM with Sigmnoid Kernel with Feature reduction Using KernelPCA: %f" % skl.metrics.f1_score(y_test, y_pred))
trainerror = accuracy_score ( y_test ,y_pred )
print ("Accuracy is: ",trainerror )
#coefs = pd.Series(cls.coef_[0], index=encoded_train.drop(['label'],axis = 1).columns)
#coefs.sort_values()
#ax = plt.subplot(2,1,2)
#coefs.plot(kind="bar")
plt.show()
# Now we strart using Distributed Estimations with feature reduction (35%)

cls  = MultinomialNB()
cls.fit(og_x_train, y_train)
y_pred = cls.predict(og_x_test)
cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders_train["label"].classes_, yticklabels=encoders_train["label"].classes_)
plt.ylabel("Real value")
plt.xlabel("Predicted value")
print ("F1 score for Naive Bayes: %f" % skl.metrics.f1_score(y_test, y_pred))
trainerror = accuracy_score ( y_test ,y_pred )
print ("Accuracy is: ",trainerror )
#coefs = pd.Series(cls.coef_[0], index=encoded_train.drop(['label'],axis = 1).columns)
#coefs.sort_values()
#ax = plt.subplot(2,1,2)
#coefs.plot(kind="bar")
plt.show()


# Now we can Use k nearest neighbours classifier Select From Model Feature Reduction Technique(60%)

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x_train, y_train)
model = SelectFromModel(lsvc, prefit=True)
xtrain_new = model.transform(x_train)
lsvc1 = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x_test, y_test)
model1 = SelectFromModel(lsvc1, prefit=True)
xtest_new = model1.transform(x_test)

cls  = KNeighborsClassifier(n_neighbors=3, algorithm = 'ball_tree')
#print (x_train)
cls.fit(xtrain_new, y_train)
y_pred = cls.predict(xtest_new)
cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders_train["label"].classes_, yticklabels=encoders_train["label"].classes_)
plt.ylabel("Real value")
plt.xlabel("Predicted value")
print ("F1 score for K nearest Neighbours: %f" % skl.metrics.f1_score(y_test, y_pred))
trainerror = accuracy_score ( y_test ,y_pred )
print ("Accuracy is: ",trainerror )
#coefs = pd.Series(cls.coef_[0], index=encoded_train.drop(['label'],axis = 1).columns)
#coefs.sort_values()
#ax = plt.subplot(2,1,2)
#coefs.plot(kind="bar")
plt.show()


#MUlti layer nueral netweok classification with back propagation(64%)

cls  = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15, 5), random_state=1)
#print (x_train)
cls.fit(x_train, y_train)
y_pred = cls.predict(x_test)
cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders_train["label"].classes_, yticklabels=encoders_train["label"].classes_)
plt.ylabel("Real value")
plt.xlabel("Predicted value")
print ("F1 score for ANN: %f" % skl.metrics.f1_score(y_test, y_pred))
trainerror = accuracy_score ( y_test ,y_pred )
print ("Accuracy is: ",trainerror )
#coefs = pd.Series(cls.coef_[0], index=encoded_train.drop(['label'],axis = 1).columns)
#coefs.sort_values()
#ax = plt.subplot(2,1,2)
#coefs.plot(kind="bar")
plt.show()

 