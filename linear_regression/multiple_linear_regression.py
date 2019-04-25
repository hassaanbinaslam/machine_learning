# Machine Learning A-Z

import pandas as pd
# importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# dataset has categorical data so we will encode it
# first step is to label the categorical data to numbers
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
# second step is to change the label into their seperate columns (one hot encoding)
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Fix Dummy Variable Trap: remove one dummy variable from the data
X = X[:, 1:]

# split the data into train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# fitting the linear regression model to our training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# now predict the test set results
y_predict = regressor.predict(X_test)

# now we will try to improve our model by checking if any input column is less significant and can be removed
# we will use backward elimination for this purpose
import statsmodels.formula.api as sm
import numpy as np
# append constant ones to the dataset
X = np.append(np.ones((50,1)).astype(int), X, axis=1)
X_optimal = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()

# index 2 has the highest P value. we will remove it and run the model again
X_optimal = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()

# index 1 has the highest P value. we will remove it and run the model again
X_optimal = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()

# index 2 has the highest P value. we will remove it and run the model again
X_optimal = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()

# index 2 has the highest P value. we will remove it and run the model again
X_optimal = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()