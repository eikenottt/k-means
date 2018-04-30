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


def fromtabtocsv(path):

    with open(path) as inf, open("seeds_dataset.csv", "w") as outf:
        regex = re.compile(r"\t+", re.IGNORECASE)
        for line in inf:
            if not line.strip(): continue
            outf.write(regex.sub(",",line))


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
