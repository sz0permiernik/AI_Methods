import sklearn as sklearn
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

    # Generating data
x, y = sklearn.datasets.make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, weights=[0.1, 0.9])
