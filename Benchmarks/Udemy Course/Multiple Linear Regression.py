#Multiple Linear Regression
# Data Preprocessing; Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values  #independent variable
Y = dataset.iloc[:, 4].values #dependent variable

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable trap
X = X[:, 1:]

#splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) #percentages for size

#Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the Test set Results
Y_pred = regressor.predict(X_test)

#Building the Optimal Model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]] #optimal team of statistically significant variables
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() #Step 2
regressor_OLS.summary()
X_opt = X[:, [0, 1, 3, 4, 5]] #optimal team of statistically significant variables
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() #Step 2
regressor_OLS.summary()
X_opt = X[:, [0, 3, 4, 5]] #optimal team of statistically significant variables
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() #Step 2
regressor_OLS.summary()
X_opt = X[:, [0, 3, 5]] #optimal team of statistically significant variables
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() #Step 2
regressor_OLS.summary()
X_opt = X[:, [0, 3]] #optimal team of statistically significant variables
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() #Step 2
regressor_OLS.summary()







