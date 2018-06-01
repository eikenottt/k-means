from kmeans import KMeansAssignment, GaussianMixtureAssignment

# Runs KMeans Clustering Algorithm

kmeans = KMeansAssignment("seeds_dataset")
kmeans.run_kmeans()

# Runs Gaussian Mixture Model Algorithm

# If you want to run another dataset you can alter the the path underneath tab-separated .txt or .csv files accepted
gmA = GaussianMixtureAssignment("seeds_dataset")
gmA.runGM()
