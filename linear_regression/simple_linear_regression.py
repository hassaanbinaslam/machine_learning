# Machine Learning A-Z

# import the libraries
import pandas as pd

# import the data set
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# split the data into train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# fitting the linear model on train data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the results from test set
y_predict = regressor.predict(X_test)

# checking the results visually - Training Set 
import matplotlib.pyplot as plt
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary vs Experience (Training Set)")
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

# checking the results visually - Test Set 
import matplotlib.pyplot as plt
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary vs Experience (Test Set)")
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()