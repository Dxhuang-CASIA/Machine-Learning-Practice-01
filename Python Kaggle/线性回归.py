import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.linear_model import  LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 线性回归 波士顿房价

bostonData = pd.read_csv(r'./data/boston.csv')
X = bostonData.iloc[:, :-1]
y = pd.DataFrame(bostonData.iloc[:, -1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 33)

ss_X = StandardScaler()
ss_y = StandardScaler()
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_y_predict = lr.predict(X_test)

sgdr = SGDRegressor()
sgdr.fit(X_train, y_train)
sgdr_y_predict = sgdr.predict(X_test)
sgdr_y_predict = sgdr_y_predict.reshape((-1, 1))

print('The value of default measurement of LinearRegression is', lr.score(X_test, y_test))
print('The value of R-squared of Linear Regression is', r2_score(y_test, lr_y_predict))
print('The mean squared error of Linear Regression is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))
print('The mean absolute error of Linear Regression is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))
print('-------------------------------------------------------------------------------------')
print('The value of default measurement of SGDRegressor is', sgdr.score(X_test, y_test))
print('The value of R-squared of SGDRegressor is', r2_score(y_test, sgdr_y_predict))
print('The mean squared error of SGDRegressor is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict)))
print('The mean absolute error of SGDRegressor is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict)))