# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path_trn = ".\\query-impala-83396.csv"
path_tst = ".\\query-impala-83397.csv"

plt.subplot(121)
df = pd.read_csv(path_trn)
df = df.sort_values(by=["score"], ascending=True)
good_ratio = df["goodratio"]
bad_ratio = df["badratio"]
ks = df["ks"]
ks_max = np.max(ks)
score = df["score"]
score_best = df[df["ks"] == ks_max]["score"].values[0]
probability_best = df[df["ks"] == ks_max]["probability"].values[0]
print("train set: %.2f, %.3f" % (score_best, probability_best))
plt.plot(score, bad_ratio, color='red', label="bad ratio")
plt.plot(score, good_ratio, color='blue', label="good ratio")
plt.plot(score, ks, color='green', label="k-s curve")
plt.title("test set ks=%.3f" % ks_max)
plt.title("train set ks=%.3f" % ks_max)
plt.ylabel("accumulate ratio")
plt.xlabel("default probability")

plt.subplot(122)
df = pd.read_csv(path_tst)
df = df.sort_values(by=["score"], ascending=True)
good_ratio = df["goodratio"]
bad_ratio = df["badratio"]
ks = df["ks"]
ks_max = np.max(ks)
score = df["score"]
score_best = df[df["ks"] == ks_max]["score"].values[0]
probability_best = df[df["ks"] == ks_max]["probability"].values[0]
print("test set: %.2f, %.3f" % (score_best, probability_best))
plt.plot(score, bad_ratio, color='red', label="bad accumulate ratio")
plt.plot(score, good_ratio, color='blue', label="good accumulate ratio")
plt.plot(score, ks, color='green', label="k-s curve")
plt.title("test set ks=%.3f" % ks_max)
plt.legend()
plt.show()
