#Simple Linear Regression
# Data Preprocessing; Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values  #independent variable
Y = dataset.iloc[:, 1].values #dependent variable

#splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0) #percentages for size

#feature scaling
# need to do this because the attributes don't have the same scale
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) #fit and transform only for the training set
X_test = sc_X.transform(X_test) #transform the test set"""

#Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train, sample_weight = 20)

#Predicting the Test Set Results
Y_pred = regressor.predict(X_test) #vector of predictions for dependent variable

#Vizualizing the Training set Results
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # Comparing real salaries to the training salaries
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Vizualizing the Test set Results
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # keep training set model here
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


