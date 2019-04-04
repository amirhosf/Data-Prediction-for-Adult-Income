# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:03:16 2019

@author: Amirhossein Forouzani
"""
import sklearn
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
from IPython.core.interactiveshell import InteractiveShell
from sklearn.svm import SVC
#-------------------------------------------------------------------------
