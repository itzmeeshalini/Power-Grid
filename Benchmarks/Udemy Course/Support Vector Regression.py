#Support Vector Regression
# Data Preprocessing; Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values  #independent variable
y = dataset.iloc[:, 2].values #dependent variable

#splitting the dataset into training and test sets
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) #percentages for size
"""
#feature scaling is important for this regressor
# need to do this because the attributes don't have the same scale
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X) 
y = sc_y.fit_transform(y)

#Fitting the Support Vector Regression to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') #rbf = gaussian in python
regressor.fit(X, y) #Creating SVR Regressor

#Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#Visualizing the Support Vector Regression Results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('SVR Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()