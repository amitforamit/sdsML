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

#Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state =0)

#Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

















