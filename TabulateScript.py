import numpy as np
from scipy.stats import shapiro
from tabulate import tabulate

realDokladnoscImplementedAdasyn = np.load('dokladnoscImplementedAdasyn()dlaRzeczywiste.npy')
realPrecyzjaImplementedAdasyn = np.load('precyzjaImplementedAdasyn()dlaRzeczywiste.npy')
realF1ImplementedAdasyn = np.load('f1ImplementedAdasyn()dlaRzeczywiste.npy')
realRecallImplementedAdasyn = np.load('recallImplementedAdasyn()dlaRzeczywiste.npy')

realDokladnoscAdasyn = np.load('dokladnoscAdasyn()dlaRzeczywiste.npy')
realPrecyzjaAdasyn = np.load('precyzjaAdasyn()dlaRzeczywiste.npy')
realF1Adasyn = np.load('f1Adasyn()dlaRzeczywiste.npy')
realRecallAdasyn = np.load('recallAdasyn()dlaRzeczywiste.npy')

realDokladnoscBorderlineSMOTE = np.load('dokladnoscBorderlineSMOTE()dlaRzeczywiste.npy')
realPrecyzjaBorderlineSMOTE = np.load('precyzjaBorderlineSMOTE()dlaRzeczywiste.npy')
realF1BorderlineSMOTE = np.load('f1BorderlineSMOTE()dlaRzeczywiste.npy')
realRecallBorderlineSMOTE = np.load('recallBorderlineSMOTE()dlaRzeczywiste.npy')

realDokladnoscSMOTE = np.load('dokladnoscSMOTE()dlaRzeczywiste.npy')
realPrecyzjaSMOTE = np.load('precyzjaSMOTE()dlaRzeczywiste.npy')
realF1SMOTE = np.load('f1SMOTE()dlaRzeczywiste.npy')
realRecallSMOTE = np.load('recallSMOTE()dlaRzeczywiste.npy')

synDokladnoscImplementedAdasyn = np.load('dokladnoscImplementedAdasyn()dlaSyntetyczne.npy')
synPrecyzjaImplementedAdasyn = np.load('precyzjaImplementedAdasyn()dlaSyntetyczne.npy')
synF1ImplementedAdasyn = np.load('f1ImplementedAdasyn()dlaSyntetyczne.npy')
synRecallImplementedAdasyn = np.load('recallImplementedAdasyn()dlaSyntetyczne.npy')

synDokladnoscAdasyn = np.load('dokladnoscADASYN()dlaSyntetyczne.npy')
synPrecyzjaAdasyn = np.load('precyzjaADASYN()dlaSyntetyczne.npy')
synF1Adasyn = np.load('f1ADASYN()dlaSyntetyczne.npy')
synRecallAdasyn = np.load('recallADASYN()dlaSyntetyczne.npy')

synDokladnoscBorderlineSMOTE = np.load('dokladnoscBorderlineSMOTE()dlaSyntetyczne.npy')
synPrecyzjaBorderlineSMOTE = np.load('precyzjaBorderlineSMOTE()dlaSyntetyczne.npy')
synF1BorderlineSMOTE = np.load('f1BorderlineSMOTE()dlaSyntetyczne.npy')
synRecallBorderlineSMOTE = np.load('recallBorderlineSMOTE()dlaSyntetyczne.npy')

synDokladnoscSMOTE = np.load('dokladnoscSMOTE()dlaSyntetyczne.npy')
synPrecyzjaSMOTE = np.load('precyzjaSMOTE()dlaSyntetyczne.npy')
synF1SMOTE = np.load('f1SMOTE()dlaSyntetyczne.npy')
synRecallSMOTE = np.load('recallSMOTE()dlaSyntetyczne.npy')

resultTableReal = [["Metryki/Metody", "Dokladnosc", "Precyzja", "F1", "Recall"],
                   ["Zaimplementowany Adasyn", np.mean(realDokladnoscImplementedAdasyn), np.mean(realPrecyzjaImplementedAdasyn),
                    np.mean(realF1ImplementedAdasyn), np.mean(realRecallImplementedAdasyn)],
                   ["Zaimportowany Adasyn", np.mean(realDokladnoscAdasyn), np.mean(realPrecyzjaAdasyn),
                    np.mean(realF1ImplementedAdasyn), np.mean(realRecallAdasyn)],
                   ["BorderlineSMOTE", np.mean(realDokladnoscBorderlineSMOTE), np.mean(realPrecyzjaBorderlineSMOTE),
                    np.mean(realF1BorderlineSMOTE), np.mean(realRecallBorderlineSMOTE)],
                   ["SMOTE", np.mean(realDokladnoscSMOTE), np.mean(realPrecyzjaSMOTE),
                    np.mean(realF1SMOTE), np.mean(realRecallSMOTE)]]

resultTableSyn = [["Metryki/Metody", "Dokladnosc", "Precyzja", "F1", "Recall"],
                  ["Zaimplementowany Adasyn", np.mean(synDokladnoscImplementedAdasyn), np.mean(synPrecyzjaImplementedAdasyn),
                   np.mean(synF1ImplementedAdasyn), np.mean(synRecallImplementedAdasyn)],
                  ["Zaimportowany Adasyn", np.mean(synDokladnoscAdasyn), np.mean(synPrecyzjaAdasyn),
                   np.mean(synF1ImplementedAdasyn), np.mean(synRecallAdasyn)],
                  ["BorderlineSMOTE", np.mean(synDokladnoscBorderlineSMOTE), np.mean(synPrecyzjaBorderlineSMOTE),
                   np.mean(synF1BorderlineSMOTE), np.mean(synRecallBorderlineSMOTE)],
                  ["SMOTE", np.mean(synDokladnoscSMOTE), np.mean(synPrecyzjaSMOTE),
                   np.mean(synF1SMOTE), np.mean(synRecallSMOTE)]]

tableReal = tabulate(resultTableReal, tablefmt='latex')
tableSyn = tabulate(resultTableSyn, tablefmt='latex')

with open('realResults.tex', 'w') as file:
    file.write(tableReal)
with open('synResults.tex', 'w') as file1:
    file1.write(tableSyn)

print(tableReal)
print(tableSyn)

statistics, realShapiroDokladnoscImplementedAdasyn = shapiro(realDokladnoscImplementedAdasyn)
statistics, realShapiroPrecyzjaImplementedAdasyn = shapiro(realPrecyzjaImplementedAdasyn)
statistics, realShapiroF1ImplementedAdasyn = shapiro(realF1ImplementedAdasyn)
statistics, realShapiroRecallImplementedAdasyn = shapiro(realRecallImplementedAdasyn)

statistics, realShapiroDokladnoscAdasyn = shapiro(realDokladnoscAdasyn)
statistics, realShapiroPrecyzjaAdasyn = shapiro(realPrecyzjaAdasyn)
statistics, realShapiroF1Adasyn = shapiro(realF1Adasyn)
statistics, realShapiroRecallAdasyn = shapiro(realRecallAdasyn)

statistics, realShapiroDokladnoscBorderlineSMOTE = shapiro(realDokladnoscBorderlineSMOTE)
statistics, realShapiroPrecyzjaBorderlineSMOTE = shapiro(realPrecyzjaBorderlineSMOTE)
statistics, realShapiroF1BorderlineSMOTE = shapiro(realF1BorderlineSMOTE)
statistics, realShapiroRecallBorderlineSMOTE = shapiro(realRecallBorderlineSMOTE)

statistics, realShapiroDokladnoscSMOTE = shapiro(realDokladnoscSMOTE)
statistics, realShapiroPrecyzjaSMOTE = shapiro(realPrecyzjaSMOTE)
statistics, realShapiroF1SMOTE = shapiro(realF1SMOTE)
statistics, realShapiroRecallSMOTE = shapiro(realRecallSMOTE)

statistics, synShapiroDokladnoscImplementedAdasyn = shapiro(synDokladnoscImplementedAdasyn)
statistics, synShapiroPrecyzjaImplementedAdasyn = shapiro(synPrecyzjaImplementedAdasyn)
statistics, synShapiroF1ImplementedAdasyn = shapiro(synF1ImplementedAdasyn)
statistics, synShapiroRecallImplementedAdasyn = shapiro(synRecallImplementedAdasyn)

statistics, synShapiroDokladnoscAdasyn = shapiro(synDokladnoscAdasyn)
statistics, synShapiroPrecyzjaAdasyn = shapiro(synPrecyzjaAdasyn)
statistics, synShapiroF1Adasyn = shapiro(synF1Adasyn)
statistics, synShapiroRecallAdasyn = shapiro(synRecallAdasyn)

statistics, synShapiroDokladnoscBorderlineSMOTE = shapiro(synDokladnoscBorderlineSMOTE)
statistics, synShapiroPrecyzjaBorderlineSMOTE = shapiro(synPrecyzjaBorderlineSMOTE)
statistics, synShapiroF1BorderlineSMOTE = shapiro(synF1BorderlineSMOTE)
statistics, synShapiroRecallBorderlineSMOTE = shapiro(synRecallBorderlineSMOTE)

statistics, synShapiroDokladnoscSMOTE = shapiro(synDokladnoscSMOTE)
statistics, synShapiroPrecyzjaSMOTE = shapiro(synPrecyzjaSMOTE)
statistics, synShapiroF1SMOTE = shapiro(synF1SMOTE)
statistics, synShapiroRecallSMOTE = shapiro(synRecallSMOTE)

resultTableRealShapiro = [["Metryki/Metody", "Dokladnosc", "Precyzja", "F1", "Recall"],
                   ["Zaimplementowany Adasyn", realShapiroDokladnoscImplementedAdasyn, realShapiroPrecyzjaImplementedAdasyn,
                    realShapiroF1ImplementedAdasyn, realShapiroRecallImplementedAdasyn],
                   ["Zaimportowany Adasyn", realShapiroDokladnoscAdasyn, realShapiroPrecyzjaAdasyn,
                    realShapiroF1Adasyn, realShapiroRecallAdasyn],
                   ["BorderlineSMOTE", realShapiroDokladnoscBorderlineSMOTE, realShapiroPrecyzjaBorderlineSMOTE,
                    realShapiroF1BorderlineSMOTE, realShapiroRecallBorderlineSMOTE],
                   ["SMOTE", realShapiroDokladnoscSMOTE, realShapiroPrecyzjaSMOTE,
                    realShapiroF1SMOTE, realShapiroRecallSMOTE]]

resultTableSynShapiro = [["Metryki/Metody", "Dokladnosc", "Precyzja", "F1", "Recall"],
                  ["Zaimplementowany Adasyn", synShapiroDokladnoscImplementedAdasyn, synShapiroPrecyzjaImplementedAdasyn,
                   synShapiroF1ImplementedAdasyn, synShapiroRecallImplementedAdasyn],
                  ["Zaimportowany Adasyn", synShapiroDokladnoscAdasyn, synShapiroPrecyzjaAdasyn,
                   synShapiroF1Adasyn, synShapiroRecallAdasyn],
                  ["BorderlineSMOTE", synShapiroDokladnoscBorderlineSMOTE, synShapiroPrecyzjaBorderlineSMOTE,
                   synShapiroF1BorderlineSMOTE, synShapiroRecallBorderlineSMOTE],
                  ["SMOTE", synShapiroDokladnoscSMOTE, synShapiroPrecyzjaSMOTE,
                   synShapiroF1SMOTE, synShapiroRecallSMOTE]]

tableRealShapiro = tabulate(resultTableRealShapiro, tablefmt='latex')
tableSynShapiro = tabulate(resultTableSynShapiro, tablefmt='latex')

with open('realShapiroResults.tex', 'w') as file2:
    file2.write(tableRealShapiro)
with open('synShapiroResults.tex', 'w') as file3:
    file3.write(tableSynShapiro)



