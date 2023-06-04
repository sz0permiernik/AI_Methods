import numpy as np
import sklearn as sklearn
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from ImplementedAdasyn import ImplementedAdasyn
from scipy import stats
import matplotlib.pyplot as plt

print("\nImplementacja metody Adasyn\n")

random_sweep = 42
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_sweep)
real = "rzeczywiste"
synt = "syntetyczne"

# Generating data
x, y = sklearn.datasets.make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_repeated=0,
                                            weights=[0.1, 0.9], random_state=random_sweep, shuffle=True)
print("Liczba próbek wygenerowanych syntetycznie (1 -> klasa większościowa, 0 -> klasa mniejszościowa):")
print(" | Przed oversamplingiem: ", Counter(y), "\n")

plt.scatter(x[:, 0], x[:, 1], c=y)
plt.title('Dane syntetyczne przed oversamplingiem')
plt.savefig('Dane_syntetyczne_przed_oversamplingiem.png')

# Real data
realData = np.loadtxt('dane_rzeczywiste.csv', delimiter=',')
real_x = realData[:, :-1]
real_y = realData[:, -1]

print("Liczba próbek z rzeczywistego zbioru danych (1 -> klasa większościowa, 0 -> klasa mniejszościowa):")
print(" | Przed oversamplingiem: ", Counter(real_y), "\n")

# Defining methods
iada = ImplementedAdasyn()
ada = ADASYN()
br = BorderlineSMOTE()
sm = SMOTE()


def testing(x, y, method, skfold, data):
    # Empty arrays for metric scores
    accArray = []
    precArray = []
    f1Array = []
    recArray = []

    method_x, method_y = method.fit_resample(x, y)
    print(f" | Po zaimportowanym {method}: ", Counter(method_y), "\n-- dla każdego folda po kolei:")

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

    print(f"-- dokładność {method} wynosi:", accArray, ", natomiast odchylenie", np.average(accArray))
    print(f"-- precyzja {method} wynosi:", precArray, ", natomiast odchylenie", np.average(precArray))
    print(f"-- f1 {method} wynosi:", f1Array, ", natomiast odchylenie", np.average(f1Array))
    print(f"-- recall {method} wynosi:", recArray, ", natomiast odchylenie", np.average(recArray))

    np.save(f'dokladnosc{method}.npy', accArray)
    np.save(f'precyzja{method}.npy', precArray)
    np.save(f'f1{method}.npy', f1Array)
    np.save(f'recall{method}.npy', recArray)

    shapiro_test = stats.shapiro(method_x)
    print(shapiro_test, "\n")

    plt.scatter(method_x[:, 0], method_x[:, 1], c=method_y)
    plt.title(f'Dane {data} po zaimportowanym {method}')
    plt.savefig(f'Dane_{data}_po_zaimportowanym_{method}.png')


# Testing methods for synthetic data
testing(x, y, iada, skf, synt)
testing(x, y, ada, skf, synt)
testing(x, y, sm, skf, synt)
testing(x, y, br, skf, synt)

# Testing methods for real data
testing(real_x, real_y, iada, skf, real)
testing(real_x, real_y, ada, skf, real)
testing(real_x, real_y, sm, skf, real)
testing(real_x, real_y, br, skf, real)
