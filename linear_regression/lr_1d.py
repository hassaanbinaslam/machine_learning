import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

# import data
for line in open("data_1d.csv"):
    col1, col2 = line.split(',')
    X.append(float(col1))
    Y.append(float(col2))

X = np.array(X)
Y = np.array(Y)

# plot the data
plt.scatter(X, Y)
plt.show()


#one way
denominator = np.mean(X**2) - np.mean(X)**2
a = ( np.mean(X*Y) - X.mean() * Y.mean() ) / denominator
b = (np.mean(Y)*np.mean(X**2) - np.mean(X) * np.mean(X*Y)) / denominator

#other way from lazy_programmer
#denominator = X.dot(X) - X.mean() * X.sum() 
#a = (X.dot(Y) - Y.mean() * X.sum()) / denominator
#b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator

#predict y = Yhat
Yhat = a*X + b
print(Yhat)

#plot Yhat
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()