import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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

linear_svr = SVR(kernel = 'linear')
linear_svr.fit(X_train, y_train)
linear_svr_y_predict = linear_svr.predict(X_test)
linear_svr_y_predict = linear_svr_y_predict.reshape((-1, 1))

poly_svr = SVR(kernel = 'poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = poly_svr.predict(X_test)
poly_svr_y_predict = poly_svr_y_predict.reshape((-1, 1))

rbf_svr = SVR(kernel = 'rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)
rbf_svr_y_predict = rbf_svr_y_predict.reshape((-1, 1))

print('R-squared value of linear SVR is', linear_svr.score(X_test, y_test))
print('The MSE of linear SVR is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print('The MAE of linear SVR is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print('-------------------------------------------------------------------------------------')
print('R-squared value of Poly SVR is', poly_svr.score(X_test, y_test))
print('The MSE of poly SVR is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print('The MAE of poly SVR is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print('-------------------------------------------------------------------------------------')
print('R-squared value of RBF SVR is', rbf_svr.score(X_test, y_test))
print('The MSE of RBF SVR is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print('The MAE of RBF SVR is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))