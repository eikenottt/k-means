import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import os
import re

from matplotlib import style
from sklearn.cluster import KMeans


class KMeansAssignment:
    """KMeansAssignment class is made for a specific assignment at UiB

    """
    _filepath = ""
    X = None

    def __init__(self, filename):
        """Constructor

        :param filename: The name of the txt file with the dataset NOT containing '.txt'

        """
        self._filepath = filename

    def _check_csv_file_(self):
        """Checks if the csv file already exists. If the file do not exist, run the function _from_tab_to_csv

        """
        if not os.path.isfile(self._filepath + ".csv"):
            self._from_tab_to_csv(self._filepath + ".txt")

    def gather_data(self):
        """Gets the data from dataset

        """
        self._check_csv_file_()
        data = pd.read_csv(self._filepath + ".csv")

        # f1 = data['length'].values
        # f2 = data['width'].values
        #
        # self.X = np.array(list(zip(f1, f2)))
        self.X = np.array(data.drop(['label'], 1))
        return self.X

    def run_kmeans(self, n_cluster=3):
        """Runs the alorithm and shows the result as a 2D graph

        :param n_cluster: Number of clusters to divide the dataset into

        """
        style.use("ggplot")
        self.gather_data()
        kmeans = KMeans(n_cluster)
        kmeans.fit(self.X)

        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_

        print("Centroid coordinates: \n", centroids, "\n")

        colors = ["y.", "c.", "r."]

        for i in range(len(self.X)):
            print("Datapoint coordinate:", self.X[i], "label:", labels[i])
            plt.plot(self.X[i][0], self.X[i][1], colors[labels[i]], markersize=10)

        plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, color='k', zorder=10)
        plt.title("Kmeans Clusters")
        plt.ylabel("Width")
        plt.xlabel("Length")
        plt.show()

    @staticmethod
    def _from_tab_to_csv(path):
        """Converts a txt file with data separated with tab(\t) to a csv file (comma separated values) and adds header
        :param path: the filename to be converted
        """
        with open(path) as inf, open("seeds_dataset.csv", "w") as outf:
            regex = re.compile(r"\t+", re.IGNORECASE)
            outf.write("area,perimeter,compactness,length,width,asym,groove,label\n")
            for line in inf:
                if not line.strip(): continue
                outf.write(regex.sub(",", line))
            inf.close()
            outf.close()

    def _create_elbow(self, min, max):
        """outputs an elbow graph figure that shows which n_cluster value is best for the spesific dataset

        :param min: the n_cluster startnumber
        :param max: the n_cluster stopnumber
        :return: a plt.figure
        """
        distorsions = []
        for k in range(min, max):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(self.X)
            distorsions.append(kmeans.inertia_)

        fig = plt.figure(figsize=(15, 5))
        plt.plot(range(min, max), distorsions)
        plt.grid(True)
        plt.title('Elbow curve')
        plt.xlabel('Clusters')
        plt.ylabel('Sum of Squared Errors')
        return fig


