# "肘部"观察法 辅助选择最佳的k
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(5.5, 6.5, (2, 10))
cluster3 = np.random.uniform(3.0, 4.0, (2, 10))

X = np.hstack((cluster1, cluster2, cluster3)).T
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

K = range(1, 10)
meandistoritions = []

for k in K:
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(X)
    meandistoritions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis = 1)) / X.shape[0])

plt.plot(K, meandistoritions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average Dispersion')
plt.title('Selecting k with Elbow Method')
plt.show()