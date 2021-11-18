import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from  sklearn.metrics import mean_squared_error, mean_absolute_error

# 波士顿房价 回归树

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

dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)
dtr_y_predict = dtr.predict(X_test)
dtr_y_predict = dtr_y_predict.reshape((-1, 1))

print('R-squared value of uni_knr:', dtr.score(X_test, y_test))
print('The MSE of uni_knr:', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict)))
print('The MAE of uni_knr:', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict)))