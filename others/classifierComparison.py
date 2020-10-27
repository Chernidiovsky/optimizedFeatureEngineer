# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


df = pd.read_csv("E:\\Download\\model_mlp.csv")
features = df.columns.tolist()
for x in ["cust_code", "closing_down"]:
    features.remove(x)
print(features)
X = df[features].values
y = df["closing_down"].astype("int64").values

dic = {
    "Linear SVM": SVC(kernel="linear", C=0.025),
    # "RBF SVM": SVC(gamma=2, C=0.1),
    "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0)),
    "Random Forest": RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1),
    "Neural Net": MLPClassifier(alpha=0.01, activation='logistic', hidden_layer_sizes=[5, 5]),
    "AdaBoost": AdaBoostClassifier(),
    # "Naive Bayes": GaussianNB(),
    # "QDA": QuadraticDiscriminantAnalysis(),
    "Logistic": LogisticRegression(solver="liblinear")
}

key = "Random Forest"
clf = dic[key]
print(key)

score_train, score_test = [], []
for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    clf.fit(X_train, y_train)
    score_train.append(clf.score(X_train, y_train))
    score_test.append(clf.score(X_test, y_test))

print(np.mean(score_train), np.max(score_train), np.min(score_train))
print(np.mean(score_test), np.max(score_test), np.min(score_test))