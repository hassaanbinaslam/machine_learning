# Machine Learning A-Z

# load the dataset
import pandas as pd
import numpy as np

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# sklearn SVR model does not include feaature scaling so we need to apply it first
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = np.ravel(sc_y.fit_transform(y.reshape(-1, 1))) 
# np.ravel has effect opposite to y.reshape


# apply to SVR model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# visualize the results
import matplotlib.pyplot as plt
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff? SVR')
plt.xlabel('positions')
plt.ylabel('salary')
plt.show()

# predict the salary at specific position: 6.5
pos = sc_X.transform(np.array([[6.5]]))
y_pred = regressor.predict(pos)
y_pred = sc_y.inverse_transform(y_pred)
