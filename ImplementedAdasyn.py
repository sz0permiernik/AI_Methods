from abc import ABC

from imblearn.over_sampling.base import BaseOverSampler
from sklearn.neighbors import NearestNeighbors
import numpy as np
from collections import Counter


class ImplementedAdasyn(BaseOverSampler):
    def __init__(self):
        super().__init__()

    def _fit_resample(self, x, y, weights):
        # Finding the minority class
        numberOfSamples = Counter(y)
        minorityClassSamples = min(numberOfSamples.values())

        # Calculating how many samples are needed to be add in minority class
        majorityClassSamples = max(numberOfSamples.values())
        samplesToAdd = int(weights * (majorityClassSamples - minorityClassSamples))

        # Creating the List for minority class
        minorityClass = np.where(y == 0)[0]
        listOfMinorityClassSamples = x[minorityClass]

        # Using Nearest Neighbors algorithm
        nbrs = NearestNeighbors().fit(listOfMinorityClassSamples)
        nbrsOfSample = nbrs.kneighbors(listOfMinorityClassSamples, return_distance=False)

        # Generating samples in minority class
        for i in range(samplesToAdd):
            randomSample = np.random.randint(len(minorityClass))
            randomSampleNBRS = np.random.choice(nbrsOfSample[randomSample])
            syntheticSample = listOfMinorityClassSamples[randomSample] + np.random.rand() * (
                    listOfMinorityClassSamples[randomSampleNBRS] - listOfMinorityClassSamples[randomSample])

            x = np.concatenate((x, np.array([syntheticSample])))
            y = np.concatenate((y, np.full(1, 0)))

        return x, y
