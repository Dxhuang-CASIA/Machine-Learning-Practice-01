import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

titanic = pd.read_csv(r'./data/titanic/fulldata.csv')
X = titanic[['Pclass', 'Age', 'Sex']]
y = titanic['Survived']
X['Age'].fillna(X['Age'].mean(), inplace = True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 33)
vec = DictVectorizer(sparse = False)

X_train = vec.fit_transform(X_train.to_dict(orient = 'record'))
X_test = vec.transform(X_test.to_dict(orient = 'record'))

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
print('The acc of RFC on testing dataset:', rfc.score(X_test, y_test))

xgbc = XGBClassifier()
xgbc.fit(X_train, y_train)
print('The acc of xgbc on testing dataset', xgbc.score(X_test, y_test))