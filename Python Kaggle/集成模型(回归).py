import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from  sklearn.metrics import mean_squared_error, mean_absolute_error

# 波士顿房价 集成回归模型

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

rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_y_predict = rfr.predict(X_test)
rfr_y_predict = rfr_y_predict.reshape((-1, 1))

etr = ExtraTreesRegressor()
etr.fit(X_train, y_train)
etr_y_predict = etr.predict(X_test)
etr_y_predict = etr_y_predict.reshape((-1, 1))

gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
gbr_y_predict = gbr.predict(X_test)
gbr_y_predict = gbr_y_predict.reshape((-1, 1))

print('R-squared value of RandForestRegressor:', rfr.score(X_test, y_test))
print('The MSE of RandForestRegressor', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict)))
print('The MAE of RandForestRegressor', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict)))
print('------------------------------------------------------')
print('R-squared value of ExtraTreesRegressor:', etr.score(X_test, y_test))
print('The MSE of ExtraTreesRegressor', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict)))
print('The MAE of ExtraTreesRegressor', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict)))
print('------------------------------------------------------')
print('R-squared value of GradientBoostingRegressor:', etr.score(X_test, y_test))
print('The MSE of GradientBoostingRegressor', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict)))
print('The MAE of GradientBoostingRegressor', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict)))