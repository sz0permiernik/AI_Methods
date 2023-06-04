import numpy as np
from tabulate import tabulate
from scipy import stats

#rzeczywiste
rdokladnoscImplementedAdasyn = np.load('dokladnoscImplementedAdasyn()dlaRzeczywiste.npy')
rprecyzjaImplementedAdasyn = np.load('precyzjaImplementedAdasyn()dlaRzeczywiste.npy')
rf1ImplementedAdasyn = np.load('f1ImplementedAdasyn()dlaRzeczywiste.npy')
rrecallImplementedAdasyn = np.load('recallImplementedAdasyn()dlaRzeczywiste.npy')

rdokladnoscAdasyn = np.load('dokladnoscAdasyn()dlaRzeczywiste.npy')
rprecyzjaAdasyn = np.load('precyzjaAdasyn()dlaRzeczywiste.npy')
rf1Adasyn = np.load('f1Adasyn()dlaRzeczywiste.npy')
rrecallAdasyn = np.load('recallAdasyn()dlaRzeczywiste.npy')

rdokladnoscBorderlineSMOTE = np.load('dokladnoscBorderlineSMOTE()dlaRzeczywiste.npy')
rprecyzjaBorderlineSMOTE = np.load('precyzjaBorderlineSMOTE()dlaRzeczywiste.npy')
rf1BorderlineSMOTE = np.load('f1BorderlineSMOTE()dlaRzeczywiste.npy')
rrecallBorderlineSMOTE = np.load('recallBorderlineSMOTE()dlaRzeczywiste.npy')

rdokladnoscSMOTE = np.load('dokladnoscSMOTE()dlaRzeczywiste.npy')
rprecyzjaSMOTE = np.load('precyzjaSMOTE()dlaRzeczywiste.npy')
rf1SMOTE = np.load('f1SMOTE()dlaRzeczywiste.npy')
rrecallSMOTE = np.load('recallSMOTE()dlaRzeczywiste.npy')

#syntetyczne
dokladnoscImplementedAdasyn = np.load('dokladnoscImplementedAdasyn()dlaSyntetyczne.npy')
precyzjaImplementedAdasyn = np.load('precyzjaImplementedAdasyn()dlaSyntetyczne.npy')
f1ImplementedAdasyn = np.load('f1ImplementedAdasyn()dlaSyntetyczne.npy')
recallImplementedAdasyn = np.load('recallImplementedAdasyn()dlaSyntetyczne.npy')

dokladnoscAdasyn = np.load('dokladnoscADASYN()dlaSyntetyczne.npy')
precyzjaAdasyn = np.load('precyzjaADASYN()dlaSyntetyczne.npy')
f1Adasyn = np.load('f1ADASYN()dlaSyntetyczne.npy')
recallAdasyn = np.load('recallADASYN()dlaSyntetyczne.npy')

dokladnoscBorderlineSMOTE = np.load('dokladnoscBorderlineSMOTE()dlaSyntetyczne.npy')
precyzjaBorderlineSMOTE = np.load('precyzjaBorderlineSMOTE()dlaSyntetyczne.npy')
f1BorderlineSMOTE = np.load('f1BorderlineSMOTE()dlaSyntetyczne.npy')
recallBorderlineSMOTE = np.load('recallBorderlineSMOTE()dlaSyntetyczne.npy')

dokladnoscSMOTE = np.load('dokladnoscSMOTE()dlaSyntetyczne.npy')
precyzjaSMOTE = np.load('precyzjaSMOTE()dlaSyntetyczne.npy')
f1SMOTE = np.load('f1SMOTE()dlaSyntetyczne.npy')
recallSMOTE = np.load('recallSMOTE()dlaSyntetyczne.npy')

resultTableReal = [["Metryki/Metody", "Dokladnosc", "Precyzja", "F1", "Recall"],
                      ["Zaimplementowany Adasyn", rdokladnoscImplementedAdasyn, rprecyzjaImplementedAdasyn,
                       rf1ImplementedAdasyn, rrecallImplementedAdasyn],
                      ["Zaimportowany Adasyn", rdokladnoscAdasyn, rprecyzjaAdasyn,
                       rf1ImplementedAdasyn, rrecallAdasyn],
                      ["BorderlineSMOTE", rdokladnoscBorderlineSMOTE, rprecyzjaBorderlineSMOTE,
                       rf1BorderlineSMOTE, rrecallBorderlineSMOTE],
                      ["SMOTE", rdokladnoscSMOTE, rprecyzjaSMOTE,
                       rf1SMOTE, rrecallSMOTE]]

resultTableSyn = [["Metryki/Metody", "Dokladnosc", "Precyzja", "F1", "Recall"],
                      ["Zaimplementowany Adasyn", dokladnoscImplementedAdasyn, precyzjaImplementedAdasyn,
                       f1ImplementedAdasyn, recallImplementedAdasyn],
                      ["Zaimportowany Adasyn", dokladnoscAdasyn, precyzjaAdasyn,
                       f1ImplementedAdasyn, recallAdasyn],
                      ["BorderlineSMOTE", dokladnoscBorderlineSMOTE, precyzjaBorderlineSMOTE,
                       f1BorderlineSMOTE, recallBorderlineSMOTE],
                      ["SMOTE", dokladnoscSMOTE, precyzjaSMOTE,
                       f1SMOTE, recallSMOTE]]

tableR = tabulate(resultTableReal, tablefmt='latex')
tableS = tabulate(resultTableSyn, tablefmt='latex')

with open('realResults.tex', 'w') as file:
    file.write(tableR)
with open('synResults.tex', 'w') as file:
    file.write(tableS)

print(tableR)
print(tableS)

#to do
#shapiro_test = stats.shapiro()
#print(shapiro_test, "\n")