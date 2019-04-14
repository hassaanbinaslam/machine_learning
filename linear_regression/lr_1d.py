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

#plot Yhat
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

#calculate r-squared
SS_res = np.square(Y-Yhat).sum()
SS_total = np.square(Y-Y.mean()).sum()
R2 = 1 - SS_res/SS_total
print("r-squared is: ", R2)

#alternate method -> lazy_programmer
d1 = Y-Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("r-squared is: ", r2)

#Note: dot product is same as sum of suqared numbers
r = np.array([1,2,3,4])
print("a.dot(a): ", r.dot(r))
print("(a**2).sum: ", (r**2).sum())