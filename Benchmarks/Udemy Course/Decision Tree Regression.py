#Decision Tree Regression Model
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

#feature scaling
# need to do this because the attributes don't have the same scale
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) #fit and transform only for the training set
X_test = sc_X.transform(X_test) #transform the test set"""

#Fitting the Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

#Predicting a new result
y_pred = regressor.predict(np.array([[6.5]]))

#Visualizing the Decision Tree Regression Results (in higher resolution)
X_grid = np.arange(min(X), max(X), 0.011)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
