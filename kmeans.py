import pandas as pd
import os
import re
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpl
from sklearn.mixture import GaussianMixture
from scipy import linalg

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


class GaussianMixtureAssignment(KMeansAssignment):
    """Makes a Gaussian Mixture Model of the dataset provided

    """
    _filepath = ""
    data = ""
    gm = None

    def __init__(self, filename):
        """Creates an instance of KMeansAssignment to gather the data

        """
        super().__init__(filename)
        self.data = super().gather_data()

    def run_gm(self):
        """Runs the code and shows the graph

        :return:
        """

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
            xpos = np.array(n_components_range) + .2 * (i - 2)
            bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                          (i + 1) * len(n_components_range)], width=.2, color=color))

        plt.xticks(n_components_range)
        plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
        plt.title('BIC Score per model')
        xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + .2 * \
               np.floor(bic.argmin() / len(n_components_range))
        plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), 'X', fontsize=20)
        spl.set_xlabel('Num of components')
        spl.legend([b[0] for b in bars], cv_types)

        # Winner
        splot = plt.subplot(2, 1, 2)
        Y_ = clf.predict(self.data)
        for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_, color_iter)):
            v, w = linalg.eigh(cov)
            if not np.any(Y_ == i):
                continue
            plt.scatter(self.data[Y_ == i, 0], self.data[Y_ == i, 1], .8, color=color)

            angle = np.arctan2(w[0][1], w[0][0])
            angle = 180. * angle / np.pi
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            ell = mpl.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(.5)
            splot.add_artist(ell)

        plt.xticks(())
        plt.yticks(())
        plt.title('Selected GMM: {} model, {} components'.format(best_gmm.covariance_type, best_gmm.n_components))
        plt.subplots_adjust(hspace=.55, bottom=0.1)
        plt.show()
        print(bic)
