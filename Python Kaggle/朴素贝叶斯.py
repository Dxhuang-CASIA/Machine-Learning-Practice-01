from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# 文本数据类别预测 朴素贝叶斯

news = fetch_20newsgroups(subset = 'all')
# print(len(news.data))
# print(news.data[500])

X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size = 0.25, random_state = 33)

vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
print(X_test.shape)
y_predict = mnb.predict(X_test)

print('The accuracy of Naive Bayes Classifier is', mnb.score(X_test, y_test))
print(classification_report(y_test, y_predict, target_names = news.target_names))