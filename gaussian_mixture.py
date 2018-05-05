import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpl
from sklearn.mixture import GaussianMixture
from scipy import linalg

from kmeans import KMeansAssignment


class GaussianMixtureAssignment(KMeansAssignment):
    """Makes a Gaussian Mixture Model of the dataset provided

    """
    _filepath = ""
    data = ""
    gm = None
    data = None

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
