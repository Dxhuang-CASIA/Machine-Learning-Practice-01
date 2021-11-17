import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# 泰坦尼克号 决策树 随机森林 梯度提升决策树

trainData = pd.read_csv(r'./data/titanic/train.csv')
testData = pd.read_csv(r'./data/titanic/test.csv')
testSurvived = pd.read_csv(r'./data/titanic/gender_submission.csv')
testData.insert(1, 'Survived', testSurvived.iloc[:, -1])

featureSelect = ['Pclass', 'Age', 'Sex']
targetSelect = ['Survived']

X_train = trainData[featureSelect]
X_test = testData[featureSelect]
y_train = trainData[targetSelect]
y_test = testData[targetSelect]

X_train['Age'].fillna(X_train['Age'].mean(), inplace = True)
X_test['Age'].fillna(X_test['Age'].mean(), inplace = True)

vec = DictVectorizer(sparse = False)
X_train = vec.fit_transform(X_train.to_dict(orient = 'record'))
X_test = vec.transform(X_test.to_dict(orient = 'record'))

# 单一决策树
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_y_pred = dtc.predict(X_test)

# 随机森林
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_y_pred = rfc.predict(X_test)

# 梯度提升决策树
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_y_pred = gbc.predict(X_test)

print('The accuracy of Decision Tree is', dtc.score(X_test, y_test))
print(classification_report(dtc_y_pred, y_test))

print('The accuracy of Random Forest Classifier is', rfc.score(X_test, y_test))
print(classification_report(rfc_y_pred, y_test))

print('The accuracy of Gradient Tree Boosting is', gbc.score(X_test, y_test))
print(classification_report(gbc_y_pred, y_test))