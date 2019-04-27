# Machine Learning A-Z

# import the dataset
import pandas as pd
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# fitting the model to dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X, y)

# visualizing the results
import numpy as np
import matplotlib.pyplot as plt
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff? Random Forrest')
plt.xlabel('positions')
plt.ylabel('salary')
plt.show()

# predict the salary at specific position: 6.5
regressor.predict(6.5)