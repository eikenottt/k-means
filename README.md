# KMeans Clustering Assignment
This project is an answer to a group assignment from UiB.
We are students and have little knowledge of the Python programming language.

This project contains two classes: `KMeansAssignment` and `GaussianMixtureAssignment`.
## KMeansAssignment Class `kmeans.py`
This class is the parent class of `GaussianMixtureAssignment`.
It is designed to take a `.txt` file with tab-separated `\t` values or a `.csv` (comma separated values) file as an input.
We have not made a way to handle exceptions, so if the input file is not of a correct type, the program will crash.

## GaussianMixtureAssignment class `kmeans.py`
This is a child class og `KMeansAssignment`.
It uses the method `gather_data()` from its `super()` class.
This class takes the same type of file as an input.

## Run the code `main.py`
To execute the code, just run the `main.py` file and enjoy.