# Data Preprocessing; Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  #independent variable
Y = dataset.iloc[:, 3].values #dependent variable

#splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) #percentages for size

#feature scaling
# need to do this because the attributes don't have the same scale
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) #fit and transform only for the training set
X_test = sc_X.transform(X_test) #transform the test set"""

