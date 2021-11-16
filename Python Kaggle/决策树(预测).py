import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# 泰坦尼克生存预测 决策树

train_data = pd.read_csv(r'./data/titanic/train.csv')
test_data = pd.read_csv(r'./data/titanic/test.csv')
true_list = pd.read_csv(r'./data/titanic/gender_submission.csv')

X_train = train_data[['Pclass', 'Age', 'Sex']]
X_test = test_data[['Pclass', 'Age', 'Sex']]
y_train = train_data['Survived']
y_test = true_list.iloc[:, -1]

X_train['Age'].fillna(X_train['Age'].mean(), inplace = True)
X_test['Age'].fillna(X_test['Age'].mean(), inplace = True)

vec = DictVectorizer(sparse = False)
X_train = vec.fit_transform(X_train.to_dict(orient = 'record'))
X_test = vec.transform(X_test.to_dict(orient = 'record'))

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_predict = dtc.predict(X_test)

print(dtc.score(X_test, y_test))
print(classification_report(y_predict, y_test, target_names = ['died', 'survived']))