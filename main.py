import sklearn as sklearn
from sklearn.datasets import make_classification

import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors

# Generating data
x, y = sklearn.datasets.make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, weights=[0.1, 0.9])

    # Adasyn
def Adasyn(x, y, weigts):
        # Finding the miniority class
    numberOfExamples = Counter(y)
    miniorityClassSamples = min(numberOfExamples.values())

        # Calculating how many examples are needed to be add in miniority class
    majorityClassSamples = max(numberOfExamples.values())
    examplesToAdd = int(weigts*(majorityClassSamples-miniorityClassSamples))

        # Creating the List for miniority class
    miniorityList = np.where(y == 0)[0]
    print(miniorityList)

        # Using Nearest Neighbors algorithm
    nbrs = NearestNeighbors().fit(x, y)
    nbrsOfExample = nbrs.kneighbors(x[miniorityList], return_distance=False)
    print(nbrsOfExample)

        # Generaiting examples in miniority class
    #for i in range(examplesToAdd):
        #to do

Adasyn(x, y, 0.9)