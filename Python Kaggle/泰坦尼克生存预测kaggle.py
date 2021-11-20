import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# 数据集准备
train = pd.read_csv(r'./data/titanic/train.csv')
test = pd.read_csv(r'./data/titanic/test.csv')

# 人工选取特征
selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']

# 划分数据集
X_train = train[selected_features]
X_test = test[selected_features]
y_train = train['Survived']

# 预处理 训练集 Age和Embarked存在缺失值 测试集 Age和Fare存在缺失值 其他数据完整
# print(X_train.info(), X_test.info())
# print(X_test['Fare'].value_counts().sum())
X_train['Embarked'].fillna('S', inplace = True) # 采用众数

X_train['Age'].fillna(X_train['Age'].mean(), inplace = True)
X_test['Age'].fillna(X_test['Age'].mean(), inplace = True)

X_test['Fare'].fillna(X_test['Fare'].mean(), inplace = True)
# print(X_train.info(), X_test.info())

# 对非数值特征向量化
dict_vec = DictVectorizer(sparse = False)
X_train = dict_vec.fit_transform(X_train.to_dict(orient = 'record'))
print(dict_vec.feature_names_)
X_test = dict_vec.transform(X_test.to_dict(orient = 'record'))

# 搭建模型
rfc = RandomForestClassifier()
xgbc = XGBClassifier()

# 交叉验证
print(cross_val_score(rfc, X_train, y_train, cv = 5).mean())
print(cross_val_score(xgbc, X_train, y_train, cv = 5).mean())

# 预测
rfc.fit(X_train, y_train)
rfc_y_pred = rfc.predict(X_test)
rfc_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': rfc_y_pred})
rfc_submission.to_csv(r'./data/titanic/rfc_submission.csv', index = False)

xgbc.fit(X_train, y_train)
xgbc_y_pred = xgbc.predict(X_test)
xgbc_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc_y_pred})
xgbc_submission.to_csv(r'./data/titanic/xgbc_submission.csv', index = False)

# 网格化搜索最优参数
params = {'max_depth':range(2, 7), 'n_estimators':range(100, 1100, 200), 'learning_rate': [0.05, 0.1, 0.25, 0.5, 1.0]}
xgbc_best = XGBClassifier()
gs = GridSearchCV(xgbc_best, params, n_jobs = -1, cv = 5, verbose = 1)
gs.fit(X_train, y_train)

print(gs.best_score_)
print(gs.best_params_)

xgbc_best_y_pred = gs.predict(X_test)
xgbc_best_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc_best_y_pred})
xgbc_best_submission.to_csv('./data/titanic/xgbc_best_submission.csv', index = False)