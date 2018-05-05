from kmeans import *

kmeans = KMeansAssignment("seeds_dataset")
kmeans.run_kmeans()

gmA = GaussianMixtureAssignment("seeds_dataset")
gmA.run_gm()
