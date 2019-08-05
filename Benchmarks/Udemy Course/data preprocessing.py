# Data Preprocessing; Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  #independent variable
Y = dataset.iloc[:, 3].values #dependent variable

#missing data
#take the mean of the columns**Weka does this for numerical values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) #0 for columns, 1 for rows
imputer = imputer.fit(X[:, 1:3]) #taking indexes 1 and 2 from the x matrix
X[:, 1:3] = imputer.transform(X[:, 1:3]) #replaces missing data by the mean of the column

#encoding categorical data
#making all nominal values numeric
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
#fit the labelencoder to the country column in the x matrix
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#dummy encoding - takes away order between the numbers (no country is greater than the other)
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
#doing the dame thing for Y
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) #percentages for size

#feature scaling
# need to do this because the attributes don't have the same scale
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) #fit and transform only for the training set
X_test = sc_X.transform(X_test) #transform the test set
