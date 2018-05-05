from kmeans import KMeansAssignment
from gaussian_mixture import GaussianMixtureAssignment

kmeans = KMeansAssignment("seeds_dataset")
kmeans.run_kmeans()

gmA = GaussianMixtureAssignment("seeds_dataset")
gmA.run_gm()
