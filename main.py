import numpy as np
import sklearn as sklearn
from sklearn.datasets import make_classification
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from ImplementedAdasyn import ImplementedAdasyn
from scipy import stats
import matplotlib.pyplot as plt
from tabulate import tabulate

print("\nImplementacja metody Adasyn\n")

random_sweep = 42
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_sweep)
real = "rzeczywiste"
synt = "syntetyczne"
tsne = TSNE()

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
    accArray = np.zeros(5)
    precArray = np.zeros(5)
    f1Array = np.zeros(5)
    recArray = np.zeros(5)

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

        accArray[i] = acc
        precArray[i] = prec
        f1Array[i] = f1
        recArray[i] = rec

    print(f"-- dokładność {method} wynosi:", accArray, ", natomiast odchylenie", np.average(accArray))
    print(f"-- precyzja {method} wynosi:", precArray, ", natomiast odchylenie", np.average(precArray))
    print(f"-- f1 {method} wynosi:", f1Array, ", natomiast odchylenie", np.average(f1Array))
    print(f"-- recall {method} wynosi:", recArray, ", natomiast odchylenie", np.average(recArray), "\n")

    resultTable = [["Metryka", "Dokladnosc", "Precyzja", "F1", "Recall"],
               ["Fold 1", accArray[0], precArray[0], f1Array[0], recArray[0]],
               ["Fold 2", accArray[1], precArray[1], f1Array[1], recArray[1]],
               ["Fold 3", accArray[2], precArray[2], f1Array[2], recArray[2]],
               ["Fold 4", accArray[3], precArray[3], f1Array[3], recArray[3]],
               ["Fold 5", accArray[4], precArray[4], f1Array[4], recArray[4]]]

    tabulateResults = tabulate(resultTable, tablefmt='latex_raw')
    with open(f'tabela{method}.tex', 'w') as file:
        file.write(tabulateResults)

    np.save(f'dokladnosc{method}.npy', accArray)
    np.save(f'precyzja{method}.npy', precArray)
    np.save(f'f1{method}.npy', f1Array)
    np.save(f'recall{method}.npy', recArray)

    #shapiro_test = stats.shapiro(method_x)
    #print(shapiro_test, "\n")

    if data == synt:
        plt.scatter(method_x[:, 0], method_x[:, 1], c=method_y)
        plt.title(f'Dane {data} po {method}')
        plt.savefig(f'Dane_{data}_po_{method}.png')

    if data == real:
        tsne_result = tsne.fit_transform(method_x)
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=method_y)
        plt.title(f'Dane {data} po {method}')
        plt.savefig(f'Dane_{data}_po_{method}.png')


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
