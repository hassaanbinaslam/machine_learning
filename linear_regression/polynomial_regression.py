# Machine Learning A-Z

# load the data set
import pandas as pd
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# fitting linear regression model to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# fitting the polynomial regression model the the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# visualizing the linear regression resutls
import matplotlib.pyplot as plt
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff? Linear Regression')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

# visualizing the polynomial regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
plt.title('Truth or Bluff? Polynomial Regression')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

# visualizing the polynomial regression results
# PLUS making the plot more smooth instead of just 10 input X points
import numpy as np
X_grid = np.arange(min(X), max(X), 0.1) # increment X with 0.1 at each step
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
X_poly = poly_reg.fit_transform(X_grid)
plt.plot(X_grid, lin_reg_2.predict(X_poly), color = 'blue')
plt.title('Truth or Bluff? Polynomial Regression')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

# now we have our model ready. we will use it to predict the salary at position 6.5
# predict using linear regression
lin_reg.predict(6.5)
# predict using polynomial regression
X_poly = poly_reg.fit_transform(6.5)
lin_reg_2.predict(X_poly)