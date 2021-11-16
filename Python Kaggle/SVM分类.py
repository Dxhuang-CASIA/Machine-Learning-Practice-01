from sklearn.datasets import load_digits # 手写数字
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# 手写数字 SVM分类模型

digits = load_digits() # 1797张 8*8

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.25, random_state = 33)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lsvc = LinearSVC(max_iter = 10000) # max_iter少了会警告
lsvc.fit(X_train, y_train)

y_predict = lsvc.predict(X_test)
print('The accuracy of Linear SVC is', lsvc.score(X_test, y_test))
print(classification_report(y_test, y_predict, target_names = digits.target_names.astype(str)))