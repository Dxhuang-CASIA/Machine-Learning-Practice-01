import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics

digits_train = pd.read_csv(r'./data/optdigits/train.csv')
digits_test = pd.read_csv(r'./data/optdigits/test.csv')
# digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
# digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header=None)

digits_train.to_csv(r'./data/optdigits/train.csv')
digits_test.to_csv(r'./data/optdigits/test.csv')

X_train = digits_train.iloc[:, :-1]
y_train = digits_train.iloc[:, -1]
X_test = digits_test.iloc[:, :-1]
y_test = digits_test.iloc[:, -1]

kmeans = KMeans(n_clusters = 10)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_test)

print(metrics.adjusted_rand_score(y_test, y_pred))