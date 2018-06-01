from kmeans import KMeansAssignment, GaussianMixtureAssignment

# Runs KMeans Clustering Algorithm

kmeans = KMeansAssignment("seeds_dataset")
kmeans.run_kmeans()

# Runs Gaussian Mixture Model Algorithm

gmA = GaussianMixtureAssignment("seeds_dataset")
gmA.runGM()