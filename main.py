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

# Generating data
x, y = sklearn.datasets.make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_repeated=0,
                                            weights=[0.1, 0.9])
print("Liczba próbek wygenerowanych syntetycznie (1 -> klasa większościowa, 2 -> klasa mniejszościowa):")
print(" | Przed oversamplingiem: ", Counter(y), "\n")

# Real data
excelData = pd.read_excel(r'data.xlsx')
realData = pd.DataFrame(excelData)
print(realData)
#real_x = realData[:, :-1]
#real_y = realData[:, -1]

# Using KNeighborsClassifier and StratifiedKFold
knc = KNeighborsClassifier(n_neighbors=5)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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
    x_train, y_train = x[train_index], y[train_index]
    x_test, y_test = x[test_index], y[test_index]

    x_train_resample, y_train_resample = implementedAdasyn._fit_resample(x_train, y_train, 0.9)
    knc.fit(x_train_resample, y_train_resample)
    y_predict = knc.predict(x[test_index])

    # Using accuracy_score, precision_score, f1_score and recall_score for implement Adasyn
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

# np.save('dokladnosc.npy', accArray)
# np.save('precyzja.npy', precArray)
# np.save('f1.npy', f1Array)
# np.save('recall.npy', recArray)

# Using imported ADASYN, SMOTE and BorderlineSMOTE for comparison
ADASYN = ADASYN()
ada_x, ada_y = ADASYN.fit_resample(x, y)
print(" | Po zaimportowanym Adasynie: ", Counter(ada_y), "\n-- dla każdego folda po kolei:")

# Empty arrays for metric scores
ada_accArray = []
ada_precArray = []
ada_f1Array = []
ada_recArray = []

# Division into training and testing sets
for i, (train_index, test_index) in enumerate(skf.split(x, y)):
    x_train, y_train = x[train_index], y[train_index]
    x_test, y_test = x[test_index], y[test_index]

    x_train_resample_ada, y_train_resample_ada = ADASYN.fit_resample(x_train, y_train)
    knc.fit(x_train_resample_ada, y_train_resample_ada)
    y_predict = knc.predict(x[test_index])

    # Using accuracy_score, precision_score, f1_score and recall_score for ADASYN
    ada_acc = accuracy_score(y[test_index], y_predict)
    ada_prec = precision_score(y[test_index], y_predict)
    ada_f1 = f1_score(y[test_index], y_predict)
    ada_rec = recall_score(y[test_index], y_predict)

    ada_accArray.append(ada_acc)
    ada_precArray.append(ada_prec)
    ada_f1Array.append(ada_f1)
    ada_recArray.append(ada_rec)

print("-- dokładność wynosi:", ada_accArray)
print("-- precyzja wynosi:", ada_precArray)
print("-- f1 wynosi:", ada_f1Array)
print("-- recall wynosi:", ada_recArray, "\n")

# np.save('dokladnoscAda.npy', ada_accArray)
# np.save('precyzjaAda.npy', ada_precArray)
# np.save('f1Ada.npy', ada_f1Array)
# np.save('recallAda.npy', ada_recArray)

sm = SMOTE()
smote_x, smote_y = sm.fit_resample(x, y)
print(" | Po zaimportowanym SMOTE: ", Counter(smote_y), "\n-- dla każdego folda po kolei:")

sm_accArray = []
sm_precArray = []
sm_f1Array = []
sm_recArray = []

# Division into training and testing sets
for i, (train_index, test_index) in enumerate(skf.split(x, y)):
    x_train, y_train = x[train_index], y[train_index]
    x_test, y_test = x[test_index], y[test_index]

    x_train_resample_sm, y_train_resample_sm = sm.fit_resample(x_train, y_train)
    knc.fit(x_train_resample_sm, y_train_resample_sm)
    y_predict = knc.predict(x[test_index])

    # Using accuracy_score, precision_score, f1_score and recall_score for SMOTE
    sm_acc = accuracy_score(y[test_index], y_predict)
    sm_prec = precision_score(y[test_index], y_predict)
    sm_f1 = f1_score(y[test_index], y_predict)
    sm_rec = recall_score(y[test_index], y_predict)

    sm_accArray.append(sm_acc)
    sm_precArray.append(sm_prec)
    sm_f1Array.append(sm_f1)
    sm_recArray.append(sm_rec)

print("-- dokładność wynosi :", sm_accArray)
print("-- precyzja wynosi :", sm_precArray)
print("-- f1 wynosi :", sm_f1Array)
print("-- recall wynosi :", sm_recArray, "\n")

# np.save('dokladnoscSm.npy', sm_accArray)
# np.save('precyzjaSm.npy', sm_precArray)
# np.save('f1Sm.npy', sm_f1Array)
# np.save('recallSm.npy', sm_recArray)

br = BorderlineSMOTE()
br_x, br_y = br.fit_resample(x, y)
print(" | Po zaimportowanym BorderlineSMOTE: ", Counter(br_y), "\n-- dla każdego folda po kolei:")

br_accArray = []
br_precArray = []
br_f1Array = []
br_recArray = []

# Division into training and testing sets
for i, (train_index, test_index) in enumerate(skf.split(x, y)):
    x_train, y_train = x[train_index], y[train_index]
    x_test, y_test = x[test_index], y[test_index]

    x_train_resample_br, y_train_resample_br = br.fit_resample(x_train, y_train)
    knc.fit(x_train_resample_br, y_train_resample_br)
    y_predict = knc.predict(x[test_index])

    # Using accuracy_score, precision_score, f1_score and recall_score for BorderlineSMOTE
    br_acc = accuracy_score(y[test_index], y_predict)
    br_prec = precision_score(y[test_index], y_predict)
    br_f1 = f1_score(y[test_index], y_predict)
    br_rec = recall_score(y[test_index], y_predict)

    br_accArray.append(br_acc)
    br_precArray.append(br_prec)
    br_f1Array.append(br_f1)
    br_recArray.append(br_rec)

print("-- dokładność wynosi :", br_accArray)
print("-- precyzja wynosi :", br_precArray)
print("-- f1 wynosi :", br_f1Array)
print("-- recall wynosi :", br_recArray)

# np.save('dokladnoscBr.npy', br_accArray)
# np.save('precyzjaBr.npy', br_precArray)
# np.save('f1Br.npy', br_f1Array)
# np.save('recallBr.npy', br_recArray)
