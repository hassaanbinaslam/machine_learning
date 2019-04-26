# Machine Learning A-Z

# import the dataset
import pandas as pd
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# fitting the decision tree to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# visualize the results
import matplotlib.pyplot as plt
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff? Decision Tree')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

# increasing the resolution of results (smoothing the curve)
import numpy as np
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff? Decision Tree')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

# predict the salary at specific position: 6.5
regressor.predict(6.5)