import numpy as np
from scipy.stats import shapiro, ttest_ind
from tabulate import tabulate

realDokladnoscImplementedAdasyn = np.load('dokladnoscImplementedAdasyn()dlaRzeczywiste.npy')
realPrecyzjaImplementedAdasyn = np.load('precyzjaImplementedAdasyn()dlaRzeczywiste.npy')
realF1ImplementedAdasyn = np.load('f1ImplementedAdasyn()dlaRzeczywiste.npy')
realRecallImplementedAdasyn = np.load('recallImplementedAdasyn()dlaRzeczywiste.npy')

realDokladnoscADASYN = np.load('dokladnoscADASYN()dlaRzeczywiste.npy')
realPrecyzjaADASYN = np.load('precyzjaADASYN()dlaRzeczywiste.npy')
realF1ADASYN = np.load('f1ADASYN()dlaRzeczywiste.npy')
realRecallADASYN = np.load('recallADASYN()dlaRzeczywiste.npy')

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

synDokladnoscADASYN = np.load('dokladnoscADASYN()dlaSyntetyczne.npy')
synPrecyzjaADASYN = np.load('precyzjaADASYN()dlaSyntetyczne.npy')
synF1ADASYN = np.load('f1ADASYN()dlaSyntetyczne.npy')
synRecallADASYN = np.load('recallADASYN()dlaSyntetyczne.npy')

synDokladnoscBorderlineSMOTE = np.load('dokladnoscBorderlineSMOTE()dlaSyntetyczne.npy')
synPrecyzjaBorderlineSMOTE = np.load('precyzjaBorderlineSMOTE()dlaSyntetyczne.npy')
synF1BorderlineSMOTE = np.load('f1BorderlineSMOTE()dlaSyntetyczne.npy')
synRecallBorderlineSMOTE = np.load('recallBorderlineSMOTE()dlaSyntetyczne.npy')

synDokladnoscSMOTE = np.load('dokladnoscSMOTE()dlaSyntetyczne.npy')
synPrecyzjaSMOTE = np.load('precyzjaSMOTE()dlaSyntetyczne.npy')
synF1SMOTE = np.load('f1SMOTE()dlaSyntetyczne.npy')
synRecallSMOTE = np.load('recallSMOTE()dlaSyntetyczne.npy')

resultTableRealMeanStd = [["Metody/Metryki", "Dokladnosc", "Precyzja", "F1", "Recall"],
                   ["Zaimplementowany Adasyn", f"{round(np.mean(realDokladnoscImplementedAdasyn), 4)} ± {round(np.std(realDokladnoscImplementedAdasyn), 4)}", f"{round(np.mean(realPrecyzjaImplementedAdasyn), 4)} ± {round(np.std(realPrecyzjaImplementedAdasyn), 4)}",
                    f"{round(np.mean(realF1ImplementedAdasyn), 4)} ± {round(np.std(realF1ImplementedAdasyn), 4)}", f"{round(np.mean(realRecallImplementedAdasyn), 4)} ± {round(np.std(realRecallImplementedAdasyn), 4)}"],
                   ["Zaimportowany Adasyn", f"{round(np.mean(realDokladnoscADASYN), 4)} ± {round(np.std(realDokladnoscADASYN), 4)}", f"{round(np.mean(realPrecyzjaADASYN), 4)} ± {round(np.std(realPrecyzjaADASYN), 4)}",
                    f"{round(np.mean(realF1ADASYN), 4)} ± {round(np.std(realF1ADASYN), 4)}", f"{round(np.mean(realRecallADASYN), 4)} ± {round(np.std(realRecallADASYN), 4)}"],
                   ["BorderlineSMOTE", f"{round(np.mean(realDokladnoscBorderlineSMOTE), 4)} ± {round(np.std(realDokladnoscBorderlineSMOTE), 4)}", f"{round(np.mean(realPrecyzjaBorderlineSMOTE), 4)} ± {round(np.std(realPrecyzjaBorderlineSMOTE), 4)}",
                    f"{round(np.mean(realF1BorderlineSMOTE), 4)} ± {round(np.std(realF1BorderlineSMOTE), 4)}", f"{round(np.mean(realRecallBorderlineSMOTE), 4)} ± {round(np.std(realRecallBorderlineSMOTE), 4)}"],
                   ["SMOTE", f"{round(np.mean(realDokladnoscSMOTE), 4)} ± {round(np.std(realDokladnoscSMOTE), 4)}", f"{round(np.mean(realPrecyzjaSMOTE), 4)} ± {round(np.std(realPrecyzjaSMOTE), 4)}",
                    f"{round(np.mean(realF1SMOTE), 4)} ± {round(np.std(realF1SMOTE), 4)}", f"{round(np.mean(realRecallSMOTE), 4)} ± {round(np.std(realRecallSMOTE), 4)}"]]

resultTableSynMeanStd = [["Metody/Metryki", "Dokladnosc", "Precyzja", "F1", "Recall"],
                  ["Zaimplementowany Adasyn", f"{round(np.mean(synDokladnoscImplementedAdasyn), 4)} ± {round(np.std(synDokladnoscImplementedAdasyn), 4)}", f"{round(np.mean(synPrecyzjaImplementedAdasyn), 4)} ± {round(np.std(synPrecyzjaImplementedAdasyn), 4)}",
                   f"{round(np.mean(synF1ImplementedAdasyn), 4)} ± {round(np.std(synF1ImplementedAdasyn), 4)}", f"{round(np.mean(synRecallImplementedAdasyn), 4)} ± {round(np.std(synRecallImplementedAdasyn), 4)}"],
                  ["Zaimportowany Adasyn", f"{round(np.mean(synDokladnoscADASYN), 4)} ± {round(np.std(realDokladnoscADASYN), 4)}", f"{round(np.mean(synPrecyzjaADASYN), 4)} ± {round(np.std(synPrecyzjaADASYN), 4)}",
                   f"{round(np.mean(synF1ADASYN), 4)} ± {round(np.std(synF1ADASYN), 4)}", f"{round(np.mean(synRecallADASYN), 4)} ± {round(np.std(synRecallADASYN), 4)}"],
                  ["BorderlineSMOTE", f"{round(np.mean(synDokladnoscBorderlineSMOTE), 4)} ± {round(np.std(synDokladnoscBorderlineSMOTE), 4)}", f"{round(np.mean(synPrecyzjaBorderlineSMOTE), 4)} ± {round(np.std(synPrecyzjaBorderlineSMOTE), 4)}",
                   f"{round(np.mean(synF1BorderlineSMOTE), 4)} ± {round(np.std(synF1BorderlineSMOTE), 4)}", f"{round(np.mean(synRecallBorderlineSMOTE), 4)} ± {round(np.std(synRecallBorderlineSMOTE), 4)}"],
                  ["SMOTE", f"{round(np.mean(synDokladnoscSMOTE), 4)} ± {round(np.std(synDokladnoscSMOTE), 4)}", f"{round(np.mean(synPrecyzjaSMOTE), 4)} ± {round(np.std(synPrecyzjaSMOTE), 4)}",
                   f"{round(np.mean(synF1SMOTE), 4)} ± {round(np.std(synF1SMOTE), 4)}", f"{round(np.mean(synRecallSMOTE), 4)} ± {round(np.std(synRecallSMOTE), 4)}"]]

tableRealMeanStd = tabulate(resultTableRealMeanStd, tablefmt='latex')
with open('tableRealMeanStd.tex', 'w') as file:
    file.write(tableRealMeanStd)

tableSynMeanStd = tabulate(resultTableSynMeanStd, tablefmt='latex')
with open('tableSynMeanStd.tex', 'w') as file1:
    file1.write(tableSynMeanStd)

print(tableRealMeanStd)
print(tableSynMeanStd)

statistics, realTtestDokladnoscImplementedAdasynVsImplementedAdasyn = ttest_ind(realDokladnoscImplementedAdasyn, realDokladnoscImplementedAdasyn)
statistics, realTtestDokladnoscImplementedAdasynVsADASYN = ttest_ind(realDokladnoscImplementedAdasyn, realDokladnoscADASYN)
statistics, realTtestDokladnoscImplementedAdasynVsBorderlineSMOTE = ttest_ind(realDokladnoscImplementedAdasyn, realDokladnoscBorderlineSMOTE)
statistics, realTtestDokladnoscImplementedAdasynVsSMOTE = ttest_ind(realDokladnoscImplementedAdasyn, realDokladnoscSMOTE)
statistics, realTtestDokladnoscADASYNVsImplementedAdasyn = ttest_ind(realDokladnoscADASYN, realDokladnoscImplementedAdasyn)
statistics, realTtestDokladnoscADASYNVsADASYN = ttest_ind(realDokladnoscADASYN, realDokladnoscADASYN)
statistics, realTtestDokladnoscADASYNVsBorderlineSMOTE = ttest_ind(realDokladnoscADASYN, realDokladnoscBorderlineSMOTE)
statistics, realTtestDokladnoscADASYNVsSMOTE = ttest_ind(realDokladnoscADASYN, realDokladnoscSMOTE)
statistics, realTtestDokladnoscBorderlineSMOTEVsImplementedAdasyn = ttest_ind(realDokladnoscBorderlineSMOTE, realDokladnoscImplementedAdasyn)
statistics, realTtestDokladnoscBorderlineSMOTEVsADASYN = ttest_ind(realDokladnoscBorderlineSMOTE, realDokladnoscADASYN)
statistics, realTtestDokladnoscBorderlineSMOTEVsBorderlineSMOTE = ttest_ind(realDokladnoscBorderlineSMOTE, realDokladnoscBorderlineSMOTE)
statistics, realTtestDokladnoscBorderlineSMOTEVsSMOTE = ttest_ind(realDokladnoscBorderlineSMOTE, realDokladnoscSMOTE)
statistics, realTtestDokladnoscSMOTEVsImplementedAdasyn = ttest_ind(realDokladnoscSMOTE, realDokladnoscImplementedAdasyn)
statistics, realTtestDokladnoscSMOTEVsADASYN = ttest_ind(realDokladnoscSMOTE, realDokladnoscADASYN)
statistics, realTtestDokladnoscSMOTEVsBorderlineSMOTE = ttest_ind(realDokladnoscSMOTE, realDokladnoscBorderlineSMOTE)
statistics, realTtestDokladnoscSMOTEVsSMOTE = ttest_ind(realDokladnoscSMOTE, realDokladnoscSMOTE)

resultTableRealTtestDokladnosc = [["Metody", "Zaimplementowany Adasyn", "Zaimportowany Adasyn", "BorderlineSMOTE", "SMOTE"],
                   ["Zaimplementowany Adasyn", round(realTtestDokladnoscImplementedAdasynVsImplementedAdasyn, 4), round(realTtestDokladnoscImplementedAdasynVsADASYN, 4),
                    round(realTtestDokladnoscImplementedAdasynVsBorderlineSMOTE, 4), round(realTtestDokladnoscImplementedAdasynVsSMOTE, 4)],
                   ["Zaimportowany Adasyn", round(realTtestDokladnoscADASYNVsImplementedAdasyn, 4), round(realTtestDokladnoscADASYNVsADASYN, 4),
                    round(realTtestDokladnoscADASYNVsBorderlineSMOTE, 4), round(realTtestDokladnoscADASYNVsSMOTE, 4)],
                   ["BorderlineSMOTE", round(realTtestDokladnoscBorderlineSMOTEVsImplementedAdasyn, 4), round(realTtestDokladnoscBorderlineSMOTEVsADASYN, 4),
                    round(realTtestDokladnoscBorderlineSMOTEVsBorderlineSMOTE, 4), round(realTtestDokladnoscBorderlineSMOTEVsSMOTE, 4)],
                   ["SMOTE", round(realTtestDokladnoscSMOTEVsImplementedAdasyn, 4), round(realTtestDokladnoscSMOTEVsADASYN, 4),
                    round(realTtestDokladnoscSMOTEVsBorderlineSMOTE, 4), round(realTtestDokladnoscSMOTEVsSMOTE, 4)]]

tableRealTtestDokladnosc = tabulate(resultTableRealTtestDokladnosc, tablefmt='latex')
with open('tableRealTtestDokladnosc.tex', 'w') as file2:
    file2.write(tableRealTtestDokladnosc)

statistics, realTtestPrecyzjaImplementedAdasynVsImplementedAdasyn = ttest_ind(realPrecyzjaImplementedAdasyn, realPrecyzjaImplementedAdasyn)
statistics, realTtestPrecyzjaImplementedAdasynVsADASYN = ttest_ind(realPrecyzjaImplementedAdasyn, realPrecyzjaADASYN)
statistics, realTtestPrecyzjaImplementedAdasynVsBorderlineSMOTE = ttest_ind(realPrecyzjaImplementedAdasyn, realPrecyzjaBorderlineSMOTE)
statistics, realTtestPrecyzjaImplementedAdasynVsSMOTE = ttest_ind(realPrecyzjaImplementedAdasyn, realPrecyzjaSMOTE)
statistics, realTtestPrecyzjaADASYNVsImplementedAdasyn = ttest_ind(realPrecyzjaADASYN, realPrecyzjaImplementedAdasyn)
statistics, realTtestPrecyzjaADASYNVsADASYN = ttest_ind(realPrecyzjaADASYN, realPrecyzjaADASYN)
statistics, realTtestPrecyzjaADASYNVsBorderlineSMOTE = ttest_ind(realPrecyzjaADASYN, realPrecyzjaBorderlineSMOTE)
statistics, realTtestPrecyzjaADASYNVsSMOTE = ttest_ind(realPrecyzjaADASYN, realPrecyzjaSMOTE)
statistics, realTtestPrecyzjaBorderlineSMOTEVsImplementedAdasyn = ttest_ind(realPrecyzjaBorderlineSMOTE, realPrecyzjaImplementedAdasyn)
statistics, realTtestPrecyzjaBorderlineSMOTEVsADASYN = ttest_ind(realPrecyzjaBorderlineSMOTE, realPrecyzjaADASYN)
statistics, realTtestPrecyzjaBorderlineSMOTEVsBorderlineSMOTE = ttest_ind(realPrecyzjaBorderlineSMOTE, realPrecyzjaBorderlineSMOTE)
statistics, realTtestPrecyzjaBorderlineSMOTEVsSMOTE = ttest_ind(realPrecyzjaBorderlineSMOTE, realPrecyzjaSMOTE)
statistics, realTtestPrecyzjaSMOTEVsImplementedAdasyn = ttest_ind(realPrecyzjaSMOTE, realPrecyzjaImplementedAdasyn)
statistics, realTtestPrecyzjaSMOTEVsADASYN = ttest_ind(realPrecyzjaSMOTE, realPrecyzjaADASYN)
statistics, realTtestPrecyzjaSMOTEVsBorderlineSMOTE = ttest_ind(realPrecyzjaSMOTE, realPrecyzjaBorderlineSMOTE)
statistics, realTtestPrecyzjaSMOTEVsSMOTE = ttest_ind(realPrecyzjaSMOTE, realPrecyzjaSMOTE)

resultTableRealTtestPrecyzja = [["Metody", "Zaimplementowany Adasyn", "Zaimportowany Adasyn", "BorderlineSMOTE", "SMOTE"],
                   ["Zaimplementowany Adasyn", round(realTtestPrecyzjaImplementedAdasynVsImplementedAdasyn, 4), round(realTtestPrecyzjaImplementedAdasynVsADASYN, 4),
                    round(realTtestPrecyzjaImplementedAdasynVsBorderlineSMOTE, 4), round(realTtestPrecyzjaImplementedAdasynVsSMOTE, 4)],
                   ["Zaimportowany Adasyn", round(realTtestPrecyzjaADASYNVsImplementedAdasyn, 4), round(realTtestPrecyzjaADASYNVsADASYN, 4),
                    round(realTtestPrecyzjaADASYNVsBorderlineSMOTE, 4), round(realTtestPrecyzjaADASYNVsSMOTE, 4)],
                   ["BorderlineSMOTE", round(realTtestPrecyzjaBorderlineSMOTEVsImplementedAdasyn, 4), round(realTtestPrecyzjaBorderlineSMOTEVsADASYN, 4),
                    round(realTtestPrecyzjaBorderlineSMOTEVsBorderlineSMOTE, 4), round(realTtestPrecyzjaBorderlineSMOTEVsSMOTE, 4)],
                   ["SMOTE", round(realTtestPrecyzjaSMOTEVsImplementedAdasyn, 4), round(realTtestPrecyzjaSMOTEVsADASYN, 4),
                    round(realTtestPrecyzjaSMOTEVsBorderlineSMOTE, 4), round(realTtestPrecyzjaSMOTEVsSMOTE, 4)]]

tableRealTtestPrecyzja = tabulate(resultTableRealTtestPrecyzja, tablefmt='latex')
with open('tableRealTtestPrecyzja.tex', 'w') as file3:
    file3.write(tableRealTtestPrecyzja)

statistics, realTtestF1ImplementedAdasynVsImplementedAdasyn = ttest_ind(realF1ImplementedAdasyn, realF1ImplementedAdasyn)
statistics, realTtestF1ImplementedAdasynVsADASYN = ttest_ind(realF1ImplementedAdasyn, realF1ADASYN)
statistics, realTtestF1ImplementedAdasynVsBorderlineSMOTE = ttest_ind(realF1ImplementedAdasyn, realF1BorderlineSMOTE)
statistics, realTtestF1ImplementedAdasynVsSMOTE = ttest_ind(realF1ImplementedAdasyn, realF1SMOTE)
statistics, realTtestF1ADASYNVsImplementedAdasyn = ttest_ind(realF1ADASYN, realF1ImplementedAdasyn)
statistics, realTtestF1ADASYNVsADASYN = ttest_ind(realF1ADASYN, realF1ADASYN)
statistics, realTtestF1ADASYNVsBorderlineSMOTE = ttest_ind(realF1ADASYN, realF1BorderlineSMOTE)
statistics, realTtestF1ADASYNVsSMOTE = ttest_ind(realF1ADASYN, realF1SMOTE)
statistics, realTtestF1BorderlineSMOTEVsImplementedAdasyn = ttest_ind(realF1BorderlineSMOTE, realF1ImplementedAdasyn)
statistics, realTtestF1BorderlineSMOTEVsADASYN = ttest_ind(realF1BorderlineSMOTE, realF1ADASYN)
statistics, realTtestF1BorderlineSMOTEVsBorderlineSMOTE = ttest_ind(realF1BorderlineSMOTE, realF1BorderlineSMOTE)
statistics, realTtestF1BorderlineSMOTEVsSMOTE = ttest_ind(realF1BorderlineSMOTE, realF1SMOTE)
statistics, realTtestF1SMOTEVsImplementedAdasyn = ttest_ind(realF1SMOTE, realF1ImplementedAdasyn)
statistics, realTtestF1SMOTEVsADASYN = ttest_ind(realF1SMOTE, realF1ADASYN)
statistics, realTtestF1SMOTEVsBorderlineSMOTE = ttest_ind(realF1SMOTE, realF1BorderlineSMOTE)
statistics, realTtestF1SMOTEVsSMOTE = ttest_ind(realF1SMOTE, realF1SMOTE)

resultTableRealTtestF1 = [["Metody", "Zaimplementowany Adasyn", "Zaimportowany Adasyn", "BorderlineSMOTE", "SMOTE"],
                   ["Zaimplementowany Adasyn", round(realTtestF1ImplementedAdasynVsImplementedAdasyn, 4), round(realTtestF1ImplementedAdasynVsADASYN, 4),
                    round(realTtestF1ImplementedAdasynVsBorderlineSMOTE, 4), round(realTtestF1ImplementedAdasynVsSMOTE, 4)],
                   ["Zaimportowany Adasyn", round(realTtestF1ADASYNVsImplementedAdasyn, 4), round(realTtestF1ADASYNVsADASYN, 4),
                    round(realTtestF1ADASYNVsBorderlineSMOTE, 4), round(realTtestF1ADASYNVsSMOTE, 4)],
                   ["BorderlineSMOTE", round(realTtestF1BorderlineSMOTEVsImplementedAdasyn, 4), round(realTtestF1BorderlineSMOTEVsADASYN, 4),
                    round(realTtestF1BorderlineSMOTEVsBorderlineSMOTE, 4), round(realTtestF1BorderlineSMOTEVsSMOTE, 4)],
                   ["SMOTE", round(realTtestF1SMOTEVsImplementedAdasyn, 4), round(realTtestF1SMOTEVsADASYN, 4),
                    round(realTtestF1SMOTEVsBorderlineSMOTE, 4), round(realTtestF1SMOTEVsSMOTE, 4)]]

tableRealTtestF1 = tabulate(resultTableRealTtestF1, tablefmt='latex')
with open('tableRealTtestF1.tex', 'w') as file4:
    file4.write(tableRealTtestF1)

statistics, realTtestRecallImplementedAdasynVsImplementedAdasyn = ttest_ind(realRecallImplementedAdasyn, realRecallImplementedAdasyn)
statistics, realTtestRecallImplementedAdasynVsADASYN = ttest_ind(realRecallImplementedAdasyn, realRecallADASYN)
statistics, realTtestRecallImplementedAdasynVsBorderlineSMOTE = ttest_ind(realRecallImplementedAdasyn, realRecallBorderlineSMOTE)
statistics, realTtestRecallImplementedAdasynVsSMOTE = ttest_ind(realRecallImplementedAdasyn, realRecallSMOTE)
statistics, realTtestRecallADASYNVsImplementedAdasyn = ttest_ind(realRecallADASYN, realRecallImplementedAdasyn)
statistics, realTtestRecallADASYNVsADASYN = ttest_ind(realRecallADASYN, realRecallADASYN)
statistics, realTtestRecallADASYNVsBorderlineSMOTE = ttest_ind(realRecallADASYN, realRecallBorderlineSMOTE)
statistics, realTtestRecallADASYNVsSMOTE = ttest_ind(realRecallADASYN, realRecallSMOTE)
statistics, realTtestRecallBorderlineSMOTEVsImplementedAdasyn = ttest_ind(realRecallBorderlineSMOTE, realRecallImplementedAdasyn)
statistics, realTtestRecallBorderlineSMOTEVsADASYN = ttest_ind(realRecallBorderlineSMOTE, realRecallADASYN)
statistics, realTtestRecallBorderlineSMOTEVsBorderlineSMOTE = ttest_ind(realRecallBorderlineSMOTE, realRecallBorderlineSMOTE)
statistics, realTtestRecallBorderlineSMOTEVsSMOTE = ttest_ind(realRecallBorderlineSMOTE, realRecallSMOTE)
statistics, realTtestRecallSMOTEVsImplementedAdasyn = ttest_ind(realRecallSMOTE, realRecallImplementedAdasyn)
statistics, realTtestRecallSMOTEVsADASYN = ttest_ind(realRecallSMOTE, realRecallADASYN)
statistics, realTtestRecallSMOTEVsBorderlineSMOTE = ttest_ind(realRecallSMOTE, realRecallBorderlineSMOTE)
statistics, realTtestRecallSMOTEVsSMOTE = ttest_ind(realRecallSMOTE, realRecallSMOTE)

resultTableRealTtestRecall = [["Metody", "Zaimplementowany Adasyn", "Zaimportowany Adasyn", "BorderlineSMOTE", "SMOTE"],
                   ["Zaimplementowany Adasyn", round(realTtestRecallImplementedAdasynVsImplementedAdasyn, 4), round(realTtestRecallImplementedAdasynVsADASYN, 4),
                    round(realTtestRecallImplementedAdasynVsBorderlineSMOTE, 4), round(realTtestRecallImplementedAdasynVsSMOTE, 4)],
                   ["Zaimportowany Adasyn", round(realTtestRecallADASYNVsImplementedAdasyn, 4), round(realTtestRecallADASYNVsADASYN, 4),
                    round(realTtestRecallADASYNVsBorderlineSMOTE, 4), round(realTtestRecallADASYNVsSMOTE, 4)],
                   ["BorderlineSMOTE", round(realTtestRecallBorderlineSMOTEVsImplementedAdasyn, 4), round(realTtestRecallBorderlineSMOTEVsADASYN, 4),
                    round(realTtestRecallBorderlineSMOTEVsBorderlineSMOTE, 4), round(realTtestRecallBorderlineSMOTEVsSMOTE, 4)],
                   ["SMOTE", round(realTtestRecallSMOTEVsImplementedAdasyn, 4), round(realTtestRecallSMOTEVsADASYN, 4),
                    round(realTtestRecallSMOTEVsBorderlineSMOTE, 4), round(realTtestRecallSMOTEVsSMOTE, 4)]]

tableRealTtestRecall = tabulate(resultTableRealTtestRecall, tablefmt='latex')
with open('tableRealTtestRecall.tex', 'w') as file5:
    file5.write(tableRealTtestRecall)

statistics, synTtestDokladnoscImplementedAdasynVsImplementedAdasyn = ttest_ind(synDokladnoscImplementedAdasyn, synDokladnoscImplementedAdasyn)
statistics, synTtestDokladnoscImplementedAdasynVsADASYN = ttest_ind(synDokladnoscImplementedAdasyn, synDokladnoscADASYN)
statistics, synTtestDokladnoscImplementedAdasynVsBorderlineSMOTE = ttest_ind(synDokladnoscImplementedAdasyn, synDokladnoscBorderlineSMOTE)
statistics, synTtestDokladnoscImplementedAdasynVsSMOTE = ttest_ind(synDokladnoscImplementedAdasyn, synDokladnoscSMOTE)
statistics, synTtestDokladnoscADASYNVsImplementedAdasyn = ttest_ind(synDokladnoscADASYN, synDokladnoscImplementedAdasyn)
statistics, synTtestDokladnoscADASYNVsADASYN = ttest_ind(synDokladnoscADASYN, synDokladnoscADASYN)
statistics, synTtestDokladnoscADASYNVsBorderlineSMOTE = ttest_ind(synDokladnoscADASYN, synDokladnoscBorderlineSMOTE)
statistics, synTtestDokladnoscADASYNVsSMOTE = ttest_ind(synDokladnoscADASYN, synDokladnoscSMOTE)
statistics, synTtestDokladnoscBorderlineSMOTEVsImplementedAdasyn = ttest_ind(synDokladnoscBorderlineSMOTE, synDokladnoscImplementedAdasyn)
statistics, synTtestDokladnoscBorderlineSMOTEVsADASYN = ttest_ind(synDokladnoscBorderlineSMOTE, synDokladnoscADASYN)
statistics, synTtestDokladnoscBorderlineSMOTEVsBorderlineSMOTE = ttest_ind(synDokladnoscBorderlineSMOTE, synDokladnoscBorderlineSMOTE)
statistics, synTtestDokladnoscBorderlineSMOTEVsSMOTE = ttest_ind(synDokladnoscBorderlineSMOTE, synDokladnoscSMOTE)
statistics, synTtestDokladnoscSMOTEVsImplementedAdasyn = ttest_ind(synDokladnoscSMOTE, synDokladnoscImplementedAdasyn)
statistics, synTtestDokladnoscSMOTEVsADASYN = ttest_ind(synDokladnoscSMOTE, synDokladnoscADASYN)
statistics, synTtestDokladnoscSMOTEVsBorderlineSMOTE = ttest_ind(synDokladnoscSMOTE, synDokladnoscBorderlineSMOTE)
statistics, synTtestDokladnoscSMOTEVsSMOTE = ttest_ind(synDokladnoscSMOTE, synDokladnoscSMOTE)

resultTableSynTtestDokladnosc = [["Metody", "Zaimplementowany Adasyn", "Zaimportowany Adasyn", "BorderlineSMOTE", "SMOTE"],
                   ["Zaimplementowany Adasyn", round(synTtestDokladnoscImplementedAdasynVsImplementedAdasyn, 4), round(synTtestDokladnoscImplementedAdasynVsADASYN, 4),
                    round(synTtestDokladnoscImplementedAdasynVsBorderlineSMOTE, 4), round(synTtestDokladnoscImplementedAdasynVsSMOTE, 4)],
                   ["Zaimportowany Adasyn", round(synTtestDokladnoscADASYNVsImplementedAdasyn, 4), round(synTtestDokladnoscADASYNVsADASYN, 4),
                    round(synTtestDokladnoscADASYNVsBorderlineSMOTE, 4), round(synTtestDokladnoscADASYNVsSMOTE, 4)],
                   ["BorderlineSMOTE", round(synTtestDokladnoscBorderlineSMOTEVsImplementedAdasyn, 4), round(synTtestDokladnoscBorderlineSMOTEVsADASYN, 4),
                    round(synTtestDokladnoscBorderlineSMOTEVsBorderlineSMOTE, 4), round(synTtestDokladnoscBorderlineSMOTEVsSMOTE, 4)],
                   ["SMOTE", round(synTtestDokladnoscSMOTEVsImplementedAdasyn, 4), round(synTtestDokladnoscSMOTEVsADASYN, 4),
                    round(synTtestDokladnoscSMOTEVsBorderlineSMOTE, 4), round(synTtestDokladnoscSMOTEVsSMOTE, 4)]]

tableSynTtestDokladnosc = tabulate(resultTableSynTtestDokladnosc, tablefmt='latex')
with open('tableSynTtestDokladnosc.tex', 'w') as file6:
    file6.write(tableSynTtestDokladnosc)

statistics, synTtestPrecyzjaImplementedAdasynVsImplementedAdasyn = ttest_ind(synPrecyzjaImplementedAdasyn, synPrecyzjaImplementedAdasyn)
statistics, synTtestPrecyzjaImplementedAdasynVsADASYN = ttest_ind(synPrecyzjaImplementedAdasyn, synPrecyzjaADASYN)
statistics, synTtestPrecyzjaImplementedAdasynVsBorderlineSMOTE = ttest_ind(synPrecyzjaImplementedAdasyn, synPrecyzjaBorderlineSMOTE)
statistics, synTtestPrecyzjaImplementedAdasynVsSMOTE = ttest_ind(synPrecyzjaImplementedAdasyn, synPrecyzjaSMOTE)
statistics, synTtestPrecyzjaADASYNVsImplementedAdasyn = ttest_ind(synPrecyzjaADASYN, synPrecyzjaImplementedAdasyn)
statistics, synTtestPrecyzjaADASYNVsADASYN = ttest_ind(synPrecyzjaADASYN, synPrecyzjaADASYN)
statistics, synTtestPrecyzjaADASYNVsBorderlineSMOTE = ttest_ind(synPrecyzjaADASYN, synPrecyzjaBorderlineSMOTE)
statistics, synTtestPrecyzjaADASYNVsSMOTE = ttest_ind(synPrecyzjaADASYN, synPrecyzjaSMOTE)
statistics, synTtestPrecyzjaBorderlineSMOTEVsImplementedAdasyn = ttest_ind(synPrecyzjaBorderlineSMOTE, synPrecyzjaImplementedAdasyn)
statistics, synTtestPrecyzjaBorderlineSMOTEVsADASYN = ttest_ind(synPrecyzjaBorderlineSMOTE, synPrecyzjaADASYN)
statistics, synTtestPrecyzjaBorderlineSMOTEVsBorderlineSMOTE = ttest_ind(synPrecyzjaBorderlineSMOTE, synPrecyzjaBorderlineSMOTE)
statistics, synTtestPrecyzjaBorderlineSMOTEVsSMOTE = ttest_ind(synPrecyzjaBorderlineSMOTE, synPrecyzjaSMOTE)
statistics, synTtestPrecyzjaSMOTEVsImplementedAdasyn = ttest_ind(synPrecyzjaSMOTE, synPrecyzjaImplementedAdasyn)
statistics, synTtestPrecyzjaSMOTEVsADASYN = ttest_ind(synPrecyzjaSMOTE, synPrecyzjaADASYN)
statistics, synTtestPrecyzjaSMOTEVsBorderlineSMOTE = ttest_ind(synPrecyzjaSMOTE, synPrecyzjaBorderlineSMOTE)
statistics, synTtestPrecyzjaSMOTEVsSMOTE = ttest_ind(synPrecyzjaSMOTE, synPrecyzjaSMOTE)

resultTableSynTtestPrecyzja = [["Metody", "Zaimplementowany Adasyn", "Zaimportowany Adasyn", "BorderlineSMOTE", "SMOTE"],
                   ["Zaimplementowany Adasyn", round(synTtestPrecyzjaImplementedAdasynVsImplementedAdasyn, 4), round(synTtestPrecyzjaImplementedAdasynVsADASYN, 4),
                    round(synTtestPrecyzjaImplementedAdasynVsBorderlineSMOTE, 4), round(synTtestPrecyzjaImplementedAdasynVsSMOTE, 4)],
                   ["Zaimportowany Adasyn", round(synTtestPrecyzjaADASYNVsImplementedAdasyn, 4), round(synTtestPrecyzjaADASYNVsADASYN, 4),
                    round(synTtestPrecyzjaADASYNVsBorderlineSMOTE, 4), round(synTtestPrecyzjaADASYNVsSMOTE, 4)],
                   ["BorderlineSMOTE", round(synTtestPrecyzjaBorderlineSMOTEVsImplementedAdasyn, 4), round(synTtestPrecyzjaBorderlineSMOTEVsADASYN, 4),
                    round(synTtestPrecyzjaBorderlineSMOTEVsBorderlineSMOTE, 4), round(synTtestPrecyzjaBorderlineSMOTEVsSMOTE, 4)],
                   ["SMOTE", round(synTtestPrecyzjaSMOTEVsImplementedAdasyn, 4), round(synTtestPrecyzjaSMOTEVsADASYN, 4),
                    round(synTtestPrecyzjaSMOTEVsBorderlineSMOTE, 4), round(synTtestPrecyzjaSMOTEVsSMOTE, 4)]]

tableSynTtestPrecyzja = tabulate(resultTableSynTtestPrecyzja, tablefmt='latex')
with open('tableSynTtestPrecyzja.tex', 'w') as file7:
    file7.write(tableSynTtestPrecyzja)

statistics, synTtestF1ImplementedAdasynVsImplementedAdasyn = ttest_ind(synF1ImplementedAdasyn, synF1ImplementedAdasyn)
statistics, synTtestF1ImplementedAdasynVsADASYN = ttest_ind(synF1ImplementedAdasyn, synF1ADASYN)
statistics, synTtestF1ImplementedAdasynVsBorderlineSMOTE = ttest_ind(synF1ImplementedAdasyn, synF1BorderlineSMOTE)
statistics, synTtestF1ImplementedAdasynVsSMOTE = ttest_ind(synF1ImplementedAdasyn, synF1SMOTE)
statistics, synTtestF1ADASYNVsImplementedAdasyn = ttest_ind(synF1ADASYN, synF1ImplementedAdasyn)
statistics, synTtestF1ADASYNVsADASYN = ttest_ind(synF1ADASYN, synF1ADASYN)
statistics, synTtestF1ADASYNVsBorderlineSMOTE = ttest_ind(synF1ADASYN, synF1BorderlineSMOTE)
statistics, synTtestF1ADASYNVsSMOTE = ttest_ind(synF1ADASYN, synF1SMOTE)
statistics, synTtestF1BorderlineSMOTEVsImplementedAdasyn = ttest_ind(synF1BorderlineSMOTE, synF1ImplementedAdasyn)
statistics, synTtestF1BorderlineSMOTEVsADASYN = ttest_ind(synF1BorderlineSMOTE, synF1ADASYN)
statistics, synTtestF1BorderlineSMOTEVsBorderlineSMOTE = ttest_ind(synF1BorderlineSMOTE, synF1BorderlineSMOTE)
statistics, synTtestF1BorderlineSMOTEVsSMOTE = ttest_ind(synF1BorderlineSMOTE, synF1SMOTE)
statistics, synTtestF1SMOTEVsImplementedAdasyn = ttest_ind(synF1SMOTE, synF1ImplementedAdasyn)
statistics, synTtestF1SMOTEVsADASYN = ttest_ind(synF1SMOTE, synF1ADASYN)
statistics, synTtestF1SMOTEVsBorderlineSMOTE = ttest_ind(synF1SMOTE, synF1BorderlineSMOTE)
statistics, synTtestF1SMOTEVsSMOTE = ttest_ind(synF1SMOTE, synF1SMOTE)

resultTableSynTtestF1 = [["Metody", "Zaimplementowany Adasyn", "Zaimportowany Adasyn", "BorderlineSMOTE", "SMOTE"],
                   ["Zaimplementowany Adasyn", round(synTtestF1ImplementedAdasynVsImplementedAdasyn, 4), round(synTtestF1ImplementedAdasynVsADASYN, 4),
                    round(synTtestF1ImplementedAdasynVsBorderlineSMOTE, 4), round(synTtestF1ImplementedAdasynVsSMOTE, 4)],
                   ["Zaimportowany Adasyn", round(synTtestF1ADASYNVsImplementedAdasyn, 4), round(synTtestF1ADASYNVsADASYN, 4),
                    round(synTtestF1ADASYNVsBorderlineSMOTE, 4), round(synTtestF1ADASYNVsSMOTE, 4)],
                   ["BorderlineSMOTE", round(synTtestF1BorderlineSMOTEVsImplementedAdasyn, 4), round(synTtestF1BorderlineSMOTEVsADASYN, 4),
                    round(synTtestF1BorderlineSMOTEVsBorderlineSMOTE, 4), round(synTtestF1BorderlineSMOTEVsSMOTE, 4)],
                   ["SMOTE", round(synTtestF1SMOTEVsImplementedAdasyn, 4), round(synTtestF1SMOTEVsADASYN, 4),
                    round(synTtestF1SMOTEVsBorderlineSMOTE, 4), round(synTtestF1SMOTEVsSMOTE, 4)]]

tableSynTtestF1 = tabulate(resultTableSynTtestF1, tablefmt='latex')
with open('tableSynTtestF1.tex', 'w') as file8:
    file8.write(tableSynTtestF1)

statistics, synTtestRecallImplementedAdasynVsImplementedAdasyn = ttest_ind(synRecallImplementedAdasyn, synRecallImplementedAdasyn)
statistics, synTtestRecallImplementedAdasynVsADASYN = ttest_ind(synRecallImplementedAdasyn, synRecallADASYN)
statistics, synTtestRecallImplementedAdasynVsBorderlineSMOTE = ttest_ind(synRecallImplementedAdasyn, synRecallBorderlineSMOTE)
statistics, synTtestRecallImplementedAdasynVsSMOTE = ttest_ind(synRecallImplementedAdasyn, synRecallSMOTE)
statistics, synTtestRecallADASYNVsImplementedAdasyn = ttest_ind(synRecallADASYN, synRecallImplementedAdasyn)
statistics, synTtestRecallADASYNVsADASYN = ttest_ind(synRecallADASYN, synRecallADASYN)
statistics, synTtestRecallADASYNVsBorderlineSMOTE = ttest_ind(synRecallADASYN, synRecallBorderlineSMOTE)
statistics, synTtestRecallADASYNVsSMOTE = ttest_ind(synRecallADASYN, synRecallSMOTE)
statistics, synTtestRecallBorderlineSMOTEVsImplementedAdasyn = ttest_ind(synRecallBorderlineSMOTE, synRecallImplementedAdasyn)
statistics, synTtestRecallBorderlineSMOTEVsADASYN = ttest_ind(synRecallBorderlineSMOTE, synRecallADASYN)
statistics, synTtestRecallBorderlineSMOTEVsBorderlineSMOTE = ttest_ind(synRecallBorderlineSMOTE, synRecallBorderlineSMOTE)
statistics, synTtestRecallBorderlineSMOTEVsSMOTE = ttest_ind(synRecallBorderlineSMOTE, synRecallSMOTE)
statistics, synTtestRecallSMOTEVsImplementedAdasyn = ttest_ind(synRecallSMOTE, synRecallImplementedAdasyn)
statistics, synTtestRecallSMOTEVsADASYN = ttest_ind(synRecallSMOTE, synRecallADASYN)
statistics, synTtestRecallSMOTEVsBorderlineSMOTE = ttest_ind(synRecallSMOTE, synRecallBorderlineSMOTE)
statistics, synTtestRecallSMOTEVsSMOTE = ttest_ind(synRecallSMOTE, synRecallSMOTE)

resultTableSynTtestRecall = [["Metody", "Zaimplementowany Adasyn", "Zaimportowany Adasyn", "BorderlineSMOTE", "SMOTE"],
                   ["Zaimplementowany Adasyn", round(synTtestRecallImplementedAdasynVsImplementedAdasyn, 4), round(synTtestRecallImplementedAdasynVsADASYN, 4),
                    round(synTtestRecallImplementedAdasynVsBorderlineSMOTE, 4), round(synTtestRecallImplementedAdasynVsSMOTE, 4)],
                   ["Zaimportowany Adasyn", round(synTtestRecallADASYNVsImplementedAdasyn, 4), round(synTtestRecallADASYNVsADASYN, 4),
                    round(synTtestRecallADASYNVsBorderlineSMOTE, 4), round(synTtestRecallADASYNVsSMOTE, 4)],
                   ["BorderlineSMOTE", round(synTtestRecallBorderlineSMOTEVsImplementedAdasyn, 4), round(synTtestRecallBorderlineSMOTEVsADASYN, 4),
                    round(synTtestRecallBorderlineSMOTEVsBorderlineSMOTE, 4), round(synTtestRecallBorderlineSMOTEVsSMOTE, 4)],
                   ["SMOTE", round(synTtestRecallSMOTEVsImplementedAdasyn, 4), round(synTtestRecallSMOTEVsADASYN, 4),
                    round(synTtestRecallSMOTEVsBorderlineSMOTE, 4), round(synTtestRecallSMOTEVsSMOTE, 4)]]

tableSynTtestRecall = tabulate(resultTableSynTtestRecall, tablefmt='latex')
with open('tableSynTtestRecall.tex', 'w') as file9:
    file9.write(tableSynTtestRecall)


