import numpy as np
import sklearn as sklearn
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from sklearn.tree import DecisionTreeClassifier


print("\nImplementacja metody Adasyn\n")

# (1) Generating data
x, y = sklearn.datasets.make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_repeated=0,
                                            weights=[0.1, 0.9])
print("Liczba próbek (1 -> klasa większościowa, 2 -> klasa mniejszościowa):")
print(" - Przed oversamplingiem: ", Counter(y), "\n")

# (2) Defining Adasyn
class imAdasyn(BaseOverSampler):
    def __init__(self):
        pass
    def _fit_resample(self, x, y, weights):
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
            syntheticSample = listOfMinorityClassSamples[randomSample] + np.random.rand() * (
                        listOfMinorityClassSamples[randomSampleNBRS] - listOfMinorityClassSamples[randomSample])

            x = np.concatenate((x, np.array([syntheticSample])))
            y = np.concatenate((y, np.full(1, 0)))

        return x, y

# (8) Division into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8)

# (9) Using implemented Adasyn
imAdasyn = imAdasyn()
im_x, im_y = imAdasyn._fit_resample(x, y, 0.9)
print(" - Po zaimplementowanym Adasynie: ", Counter(im_y))

# Using accuracy_score, precision_score, f1_score and recall_score for implement Adasyn
X_train_resample, y_train_resample = imAdasyn._fit_resample(X_train, y_train, 0.9)

clf = DecisionTreeClassifier(random_state=8)
clf.fit(X_train_resample, y_train_resample)

y_predict = clf.predict(X_test)

acc = accuracy_score(y_test, y_predict)
prec = precision_score(y_test, y_predict, average='weighted')
f1 = f1_score(y_test, y_predict, average='weighted')
rec = recall_score(y_test, y_predict, average='weighted')

print("-- dokładność wynosi :", acc)
print("-- precyzja wynosi :", prec)
print("-- f1 wynosi :", f1)
print("-- recall wynosi :", rec, "\n")

#np.save('dokladnosc.npy', acc)
#np.save('precyzja.npy', prec)
#np.save('f1.npy', f1)
#np.save('recall.npy', rec)

# (10) Using imported ADASYN, SMOTE and BorderlineSMOTE for comparison
ADASYN = ADASYN()
ada_x, ada_y = ADASYN.fit_resample(x, y)
print(" - Po zaimportowanym Adasynie: ", Counter(ada_y))

# Using accuracy_score, precision_score, f1_score and recall_score for ADASYN
X_train_resample_ada, y_train_resample_ada = ADASYN.fit_resample(X_train, y_train)

ada_clf = DecisionTreeClassifier(random_state=8)
ada_clf.fit(X_train_resample_ada, y_train_resample_ada)

ada_y_predict = ada_clf.predict(X_test)

ada_acc = accuracy_score(y_test, ada_y_predict)
ada_prec = precision_score(y_test, ada_y_predict, average='weighted')
ada_f1 = f1_score(y_test, y_predict, average='weighted')
ada_rec = recall_score(y_test, y_predict, average='weighted')

print("-- dokładność wynosi :", ada_acc)
print("-- precyzja wynosi :", ada_prec)
print("-- f1 wynosi :", ada_f1)
print("-- recall wynosi :", ada_rec, "\n")

#np.save('dokladnoscAda.npy', ada_acc)
#np.save('precyzjaAda.npy', ada_prec)
#np.save('f1Ada.npy', ada_f1)
#np.save('recallAda.npy', ada_rec)

sm = SMOTE()
smote_x, smote_y = sm.fit_resample(x, y)
print(" - Po zaimportowanym SMOTE: ", Counter(smote_y))

# Using accuracy_score, precision_score, f1_score and recall_score for SMOTE
X_train_resample_sm, y_train_resample_sm = sm.fit_resample(X_train, y_train)

sm_clf = DecisionTreeClassifier(random_state=8)
sm_clf.fit(X_train_resample_sm, y_train_resample_sm)

sm_y_predict = sm_clf.predict(X_test)

sm_acc = accuracy_score(y_test, sm_y_predict)
sm_prec = precision_score(y_test, sm_y_predict, average='weighted')
sm_f1 = f1_score(y_test, y_predict, average='weighted')
sm_rec = recall_score(y_test, y_predict, average='weighted')

print("-- dokładność wynosi :", sm_acc)
print("-- precyzja wynosi :", sm_prec)
print("-- f1 wynosi :", sm_f1)
print("-- recall wynosi :", sm_rec, "\n")

#np.save('dokladnoscSm.npy', sm_acc)
#np.save('precyzjaSm.npy', sm_prec)
#np.save('f1Sm.npy', sm_f1)
#np.save('recallSm.npy', sm_rec)

br = BorderlineSMOTE()
br_x, br_y = br.fit_resample(x, y)
print(" - Po zaimportowanym BorderlineSMOTE: ", Counter(br_y))

# Using accuracy_score, precision_score, f1_score and recall_score for BorderlineSMOTE
X_train_resample_br, y_train_resample_br = br.fit_resample(X_train, y_train)

br_clf = DecisionTreeClassifier(random_state=8)
br_clf.fit(X_train_resample_br, y_train_resample_br)

br_y_predict = br_clf.predict(X_test)

br_acc = accuracy_score(y_test, br_y_predict)
br_prec = precision_score(y_test, br_y_predict, average='weighted')
br_f1 = f1_score(y_test, y_predict, average='weighted')
br_rec = recall_score(y_test, y_predict, average='weighted')

print("-- dokładność wynosi :", br_acc)
print("-- precyzja wynosi :", br_prec)
print("-- f1 wynosi :", br_f1)
print("-- recall wynosi :", br_rec, "\n")

#np.save('dokladnoscBr.npy', br_acc)
#np.save('precyzjaBr.npy', br_acc)
#np.save('f1Br.npy', br_acc)
#np.save('recallBr.npy', br_acc)

