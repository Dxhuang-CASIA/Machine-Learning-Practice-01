import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

# 分类问题 逻辑回归

data = pd.read_csv('./data/breast_cancer_prediction_data.csv')
data = data.replace(to_replace = '?', value = np.nan)
data = data.dropna(how = 'any')

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size = 0.25, random_state = 33)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lr = LogisticRegression() # 极大似然估计
sgdc = SGDClassifier() # SGD

lr.fit(X_train, y_train)
lr_y_predict = lr.predict(X_test)

sgdc.fit(X_train, y_train)
sgdc_y_predict = sgdc.predict(X_test)

print('Accuracy of LR Classifier:', lr.score(X_test, y_test))
print(classification_report(y_test, lr_y_predict, target_names = ['Benign', 'Maligant']))
print('Accuracy of SGD Classifer:', sgdc.score(X_test, y_test))
print(classification_report(y_test, sgdc_y_predict, target_names = ['Benign', 'Maligant']))