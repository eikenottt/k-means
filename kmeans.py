import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import style
from sklearn import preprocessing
from sklearn.cluster import KMeans

from fromtxttoarray import fromtxttodata

style.use("ggplot")

data = pd.read_csv("seeds_dataset.csv")

f1 = data['area'].values
f2 = data['groove'].values

X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c="black", s=7)
plt.show()

# X = np.array(X)

# y = np.array(X[:, 7])
#
# print(y)
#
# # print(len(data))
#
# # data = [[1, 2, 4],
# #         [5, 8, 6],
# #         [1.5, 1.8, 1.6],
# #         [8, 8, 7],
# #         [1, 0.6, 8],
# #         [9, 11, 8]]
#
# # X = np.array(data)
#
# kmeans = KMeans(n_clusters=2)
# kmeans.fit(X)
#
# # plt.scatter(X[:, 0], X[:, 6])
# # plt.show()
#
# centroids = kmeans.cluster_centers_
# labels = kmeans.labels_
# #
# print(centroids)
# print(labels)
# #
# colors = ["b.", "r.", "y.", "c.", "m.", "w.", "g.", "k."]
#
# for i in range(len(X)):
#     print("coordiunate:", X[i], "label:", labels[i])
#     plt.plot(X[i][0], X[i][6], colors[labels[i]], markersize=10)
#
# plt.scatter(centroids[:, 0], centroids[:, 6], marker="x", s=150, linewidths=5, zorder=10)
#
# plt.show()
