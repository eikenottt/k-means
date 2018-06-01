import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os
import re
import itertools

from matplotlib import style
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class KMeansAssignment:
    _filepath = ""
    data = None

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

        self.data = np.array(data.drop(['label'], 1))
        return self.data

    def run_kmeans(self, n_cluster=3):
        """Runs the alorithm and shows the result as a 2D graph

            :param n_cluster: Number of clusters to divide the dataset into

            """
        style.use("ggplot")
        self.gather_data()
        kmeans = KMeans(n_cluster)
        kmeans.fit(self.data)

        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_

        print("Centroid coordinates: \n", centroids, "\n")

        colors = ["y.", "c.", "r."]

        for i in range(len(self.data)):
            print("Datapoint coordinate:", self.data[i], "label:", labels[i])
            plt.plot(self.data[i][0], self.data[i][1], colors[labels[i]], markersize=10)

        plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, color='k', zorder=10)
        plt.title("Kmeans Clusters")
        plt.ylabel("Width")
        plt.xlabel("Length")
        plt.show()

    def _from_tab_to_csv(self, path):
        """Converts a txt file with data separated with tab(\t) to a csv file (comma separated values) and adds header
            :param path: the filename to be converted
            """
        with open(path) as source_file, open("seeds_dataset.csv", "w") as new_file:
            regex = re.compile(r"\t+", re.IGNORECASE)
            new_file.write("area,perimeter,compactness,length,width,asym,groove,label\n")
            for line in source_file:
                if not line.strip(): continue
                new_file.write(regex.sub(",", line))
            source_file.close()
            new_file.close()

    def _create_elbow(self, min, max):
        distorsions = []
        for cluster_count in range(min, max):
            kmeans = KMeans(n_clusters=cluster_count)
            kmeans.fit(self.data)
            distorsions.append(kmeans.inertia_)

        fig = plt.figure(figsize=(15, 5))
        plt.plot(range(min, max), distorsions)
        plt.grid(True)
        plt.title('Elbow curve')
        plt.xlabel('Clusters')
        plt.ylabel('Sum of Squared Errors')
        return fig


class GaussianMixtureAssignment:
    gaussian_model = None
    data = None

    def __init__(self, dataset):
        kmeans_ = KMeansAssignment(dataset)
        self.data = kmeans_.gather_data()

    def runGM(self):

        global best_gmm
        lowest_bic = np.infty
        bic = []
        n_components_range = range(1, 7)
        cv_types = ['spherical', 'tied', 'diag', 'full']
        for cv_type in cv_types:
            for n_components in n_components_range:
                gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type)
                gmm.fit(self.data)
                bic.append(gmm.bic(self.data))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm = gmm

        bic = np.array(bic)
        color_iter = itertools.cycle(['navy', 'green', 'purple', 'cornflowerblue'])
        clf = best_gmm
        bars = []

        # BIC model
        spl = plt.subplot(2, 1, 1)
        for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
            center_point = np.array(n_components_range) + .2 * (i - 2)
            bars.append(plt.bar(center_point, bic[i * len(n_components_range):
                                          (i + 1) * len(n_components_range)], width=.2, color=color))

        plt.xticks(n_components_range)
        plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
        plt.title('BIC Score per model')
        center_point = np.mod(bic.argmin(), len(n_components_range)) + .65 + \
               .2 * np.floor(bic.argmin() / len(n_components_range))
        plt.text(center_point, bic.min() * 0.97 + .03 * bic.max(), 'X', fontsize=20)
        spl.set_xlabel('Num of components')
        spl.legend([b[0] for b in bars], cv_types)

        # Winner
        splot = plt.subplot(2, 1, 2)
        best_prediction = clf.predict(self.data)
        for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_, color_iter)):
            v, w = linalg.eigh(cov)
            if not np.any(best_prediction == i):
                continue
            plt.scatter(self.data[best_prediction == i, 0], self.data[best_prediction == i, 1], .8, color=color)

            angle = np.arctan2(w[0][1], w[0][0])
            angle = 180. * angle / np.pi
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            ell = patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(.5)
            splot.add_artist(ell)

        plt.xticks(())
        plt.yticks(())
        plt.title('Selected GMM: {} model, {} components'.format(best_gmm.covariance_type, best_gmm.n_components))
        plt.subplots_adjust(hspace=.55, bottom=0.1)
        plt.show()
