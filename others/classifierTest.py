# -*- coding: utf-8 -*-
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

import pandas as pd


clf = MLPClassifier(activation='logistic',
                    hidden_layer_sizes=[5, 5],
                    alpha=0.01)

df = pd.read_csv("E:\\Download\\model_mlp.csv")
features = df.columns.tolist()
for x in ["cust_code", "closing_down"]:
    features.remove(x)
print(features)
X = df[features].values
y = df["closing_down"].astype("int64").values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

clf.fit(X_train, y_train)
score = clf.score(X_train, y_train)
print(score)
score = clf.score(X_test, y_test)
print(score)
