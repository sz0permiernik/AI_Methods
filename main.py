import columns as columns
import numpy as np
import sklearn as sklearn
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from ImplementedAdasyn import ImplementedAdasyn
import pandas as pd

print("\nImplementacja metody Adasyn\n")

random_sweep = 42
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_sweep)

# Generating data
x, y = sklearn.datasets.make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_repeated=0,
                                            weights=[0.1, 0.9], random_state=random_sweep, shuffle=True)
print("Liczba próbek wygenerowanych syntetycznie (1 -> klasa większościowa, 0 -> klasa mniejszościowa):")
print(" | Przed oversamplingiem: ", Counter(y), "\n")

# Real data
excelData = pd.read_excel('data.xlsx')
real_y = excelData['Class']
real_x = excelData.drop('Class', axis=1)

real_y = real_y.drop(columns=0, axis=1)

print(real_x, real_y)
print("Liczba próbek z rzeczywistego zbioru danych (1 -> klasa większościowa, 0 -> klasa mniejszościowa):")
print(" | Przed oversamplingiem: ", Counter(real_y), "\n")

# Using implemented Adasyn
implementedAdasyn = ImplementedAdasyn()
im_x, im_y = implementedAdasyn._fit_resample(x, y, 0.9)
print(" | Po zaimplementowanym Adasynie: ", Counter(im_y), "\n-- dla każdego folda po kolei:")

# Empty arrays for metric scores
accArray = []
precArray = []
f1Array = []
recArray = []

# Division into training and testing sets
for i, (train_index, test_index) in enumerate(skf.split(x, y)):
    knc = KNeighborsClassifier(n_neighbors=5)

    x_train, y_train = x[train_index], y[train_index]
    # x_test, y_test = x[test_index], y[test_index]

    x_train_resample, y_train_resample = implementedAdasyn._fit_resample(x_train, y_train, 0.9)
    knc.fit(x_train_resample, y_train_resample)
    y_predict = knc.predict(x[test_index])

    # Using accuracy_score, precision_score, f1_score and recall_score for implemented Adasyn
    acc = accuracy_score(y[test_index], y_predict)
    prec = precision_score(y[test_index], y_predict)
    f1 = f1_score(y[test_index], y_predict)
    rec = recall_score(y[test_index], y_predict)

    accArray.append(acc)
    precArray.append(prec)
    f1Array.append(f1)
    recArray.append(rec)

print("-- dokładność wynosi:", accArray)
print("-- precyzja wynosi:", precArray)
print("-- f1 wynosi:", f1Array)
print("-- recall wynosi:", recArray, "\n")

np.save('dokladnosc.npy', accArray)
np.save('precyzja.npy', precArray)
np.save('f1.npy', f1Array)
np.save('recall.npy', recArray)

# Using imported ADASYN, SMOTE and BorderlineSMOTE for comparison
ada = ADASYN()
br = BorderlineSMOTE()
sm = SMOTE()


def testing(x, y, method, skfold):
    # Empty arrays for metric scores
    accArray = []
    precArray = []
    f1Array = []
    recArray = []

    ob_x, ob_y = method.fit_resample(x, y)
    print(f" | Po zaimportowanym {method}: ", Counter(ob_y), "\n-- dla każdego folda po kolei:")

    # Division into training and testing sets
    for i, (train_index, test_index) in enumerate(skfold.split(x, y)):
        # Using KNeighborsClassifier
        knc = KNeighborsClassifier(n_neighbors=5)

        x_train, y_train = x[train_index], y[train_index]
        # x_test, y_test = x[test_index], y[test_index]

        x_train_resample, y_train_resample = method.fit_resample(x_train, y_train)
        knc.fit(x_train_resample, y_train_resample)
        y_predict = knc.predict(x[test_index])

        # Using accuracy_score, precision_score, f1_score and recall_score
        acc = accuracy_score(y[test_index], y_predict)
        prec = precision_score(y[test_index], y_predict)
        f1 = f1_score(y[test_index], y_predict)
        rec = recall_score(y[test_index], y_predict)

        accArray.append(acc)
        precArray.append(prec)
        f1Array.append(f1)
        recArray.append(rec)

    print(f"-- dokładność {method} wynosi:", accArray)
    print(f"-- precyzja {method} wynosi:", precArray)
    print(f"-- f1 {method} wynosi:", f1Array)
    print(f"-- recall {method} wynosi:", recArray, "\n")

    np.save(f'dokladnosc{method}.npy', accArray)
    np.save(f'precyzja{method}.npy', precArray)
    np.save(f'f1{method}.npy', f1Array)
    np.save(f'recall{method}.npy', recArray)


# Testing methods for synthetic data
testing(x, y, ada, skf)
testing(x, y, sm, skf)
testing(x, y, br, skf)

# Testing methods for real data
#testing(real_x, real_y, ada, skf)
#testing(real_x, real_y, sm, skf)
#testing(real_x, real_y, br, skf)