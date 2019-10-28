import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# load the data
X = []
Y = []
for line in open("data_2d.csv"):
    x1, x2, y = line.split(',')
    X.append([1, float(x1), float(x2)])
    Y.append(float(y))

# convert x and y to numpy arrays
X = np.array(X)
Y = np.array(Y)

# now plot the data to see what the data looks like
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(X[:,0], X[:,1], Y)
#plt.show()

# calculating weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

#calculate r-squared
SS_res = np.square(Y-Yhat).sum()
SS_total = np.square(Y-Y.mean()).sum()
R2 = 1 - SS_res/SS_total
print("r-squared is: ", R2) 