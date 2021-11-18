import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from  sklearn.metrics import mean_squared_error, mean_absolute_error

# 波士顿房价 KNN回归

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

uni_knr = KNeighborsRegressor(weights = 'uniform')
uni_knr.fit(X_train, y_train)
uni_knr_y_predict = uni_knr.predict(X_test)

dis_knr = KNeighborsRegressor(weights = 'distance')
dis_knr.fit(X_train, y_train)
dis_knr_y_predict = dis_knr.predict(X_test)

print('R-squared value of uni_knr:', uni_knr.score(X_test, y_test))
print('The MSE of uni_knr:', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict)))
print('The MAE of uni_knr:', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict)))
print('-----------------------------------------')
print('R-squared value of dis_knr:', dis_knr.score(X_test, y_test))
print('The MSE of dis_knr:', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict)))
print('The MAE of dis_knr:', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict)))