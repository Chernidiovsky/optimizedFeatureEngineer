# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import sys

path_trn = ".\\query-impala-83396.csv"
path_tst = ".\\query-impala-83397.csv"

plt.subplot(121)
df = pd.read_csv(path_trn)
df = df.sort_values(by=["label"])
y1 = np.array(df["label"])
y2 = np.array(df["probability"])
fpr, tpr, thresholds_trn = roc_curve(y1, y2, pos_label=1)
area = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', label="lr model")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label="random model")
plt.plot([0, 1], [1, 1], color='red', linestyle='--', label="perfect model")
plt.title("train set auc=%.3f" % area)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.subplot(122)
df = pd.read_csv(path_tst)
df = df.sort_values(by=["label"])
y1 = np.array(df["label"])
y2 = np.array(df["probability"])
fpr, tpr, thresholds_tst = roc_curve(y1, y2, pos_label=1)
area = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', label="lr model")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label="random model")
plt.plot([0, 1], [1, 1], color='red', linestyle='--', label="ideal model")
plt.title("test set auc=%.3f" % area)
plt.legend()
plt.show()
