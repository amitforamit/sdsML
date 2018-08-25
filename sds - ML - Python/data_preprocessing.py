# Data Preprocessing

# Importing the Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('C:/Users/Admin/Desktop/Machine Learning/SDS- ML - A2Z/Machine Learning A-Z/Part 1 - Data Preprocessing/Data_Preprocessing/Data.csv')

#creating matrix of features
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#taking care of missing data

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values ='NaN',strategy = 'mean',axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3]= imputer.transform(X[:,1:3])