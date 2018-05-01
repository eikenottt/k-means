import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lxml
import re
from sklearn.cluster import KMeans


def fromtxttodata(path):
    data = []

    with open(path) as f:
        for line in f:
            try:
                temp = [float(x) for x in re.compile("\t|,").split(line)]
            except ValueError as e:
                continue
            data.append(temp)

    X = np.array(data)

    f.close()

    return data


# fromtabtocsv("seeds_dataset.txt")

#
# sse = []
# maxK = 15
#
# print()
# X = np.array(fromtxttodata("seeds_dataset.txt"))
#
# for k in range(1, 15):
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(X)
#     mean = kmeans.cluster_centers_
