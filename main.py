import numpy as np
import sklearn as sklearn
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
from collections import Counter

print("\nImplementacja metody Adasyn\n")

# (1) Generating data
x, y = sklearn.datasets.make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_repeated=0,
                                            weights=[0.1, 0.9])
print("Liczba próbek (1 -> klasa większościowa, 2 -> klasa mniejszościowa):")
print(" - przed oversamplingiem: ", Counter(y))

# (2) Defining Adasyn
def Adasyn(x, y, weights):

    # (3) Finding the minority class
    numberOfSamples = Counter(y)
    minorityClassSamples = min(numberOfSamples.values())

    # (4) Calculating how many samples are needed to be add in minority class
    majorityClassSamples = max(numberOfSamples.values())
    samplesToAdd = int(weights * (majorityClassSamples - minorityClassSamples))

    # (5) Creating the List for minority class
    minorityClass = np.where(y == 0)[0]
    listOfMinorityClassSamples = x[minorityClass]

    # (6) Using Nearest Neighbors algorithm
    nbrs = NearestNeighbors().fit(listOfMinorityClassSamples)
    nbrsOfSample = nbrs.kneighbors(listOfMinorityClassSamples, return_distance=False)

    # (7) Generating samples in minority class
    for i in range(samplesToAdd):
        randomSample = np.random.randint(len(minorityClass))
        randomSampleNBRS = np.random.choice(nbrsOfSample[randomSample])
        syntheticSample = listOfMinorityClassSamples[randomSample] + np.random.rand() * (listOfMinorityClassSamples[randomSampleNBRS] - listOfMinorityClassSamples[randomSample])

        x = np.concatenate((x, np.array([syntheticSample])))
        y = np.concatenate((y, np.full(1, 0)))

    return x, y

x_ada, y_ada = Adasyn(x, y, 0.9)

print(" - po oversamplingu: ", Counter(y_ada))
