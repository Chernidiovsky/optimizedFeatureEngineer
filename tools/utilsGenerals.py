import pandas as pd
import numpy as np
import tools.utilsPlot as tuPlot
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_curve, auc


def getTableDict(varTables, labelTable):
    if not isinstance(labelTable, str):
        raise Exception("标签表表名格式有误，应为str")
    if isinstance(varTables, str):
        varTables = [varTables]
    elif isinstance(varTables, list):
        pass
    else:
        raise Exception("变量表表名格式有误，应为str或list")
    if len(varTables) > len(set(varTables)):
        raise Exception("变量表表名有重复")
    if labelTable in varTables:
        raise Exception("变量表中含有标签表")
    tableDict = {labelTable: "y"}
    for i, t in enumerate(varTables):
        tableDict[t] = "v%s" % (i + 1)
    return tableDict


def groupByKey(df, key):
    data = []
    for row in df.values.tolist():
        data.append([row[0], row[1:]])
    df = pd.DataFrame(data=data, columns=[key, "values"])
    data = df.groupby(key).agg({"values": lambda a: list(a)}).reset_index().values
    df = pd.DataFrame(data=data, columns=[key, "values"])
    return df


def checkColType(df, col):
    if df[col].dtype.kind == 'f' or df[col].dtype.kind == 'i':
        return df
    if df[col].dtype == "object":
        try:
            df[col] = df[col].astype("int32")
        except:
            try:
                df[col] = df[col].astype("int64")
            except:
                try:
                    df[col] = df[col].astype(float)
                except:
                    return pd.DataFrame()
        return df
    return pd.DataFrame()


def labels2Bound(labels):
    bounds = labels.split(" ~ ")
    return [float(bounds[0]), float(bounds[1])]


def mapScore(value, binning, retWoe=False):
    roundNum = 0
    for i in range(len(binning)):
        labels = binning.iloc[0].at["inputvalue"]
        tmp = len(str(labels).split('.')[-1])
        if tmp > roundNum:
            roundNum = tmp
        labels = binning.iloc[0].at["inputvalue_shift"]
        tmp = len(str(labels).split('.')[-1])
        if tmp > roundNum:
            roundNum = tmp
    value = round(value, roundNum)
    result = binning[(binning['inputvalue'] <= value) & (binning['inputvalue_shift'] >= value)]
    if result.empty:  # 空值分配到id最高的一组，即means最低的组，表示其和目标变量无关
        idMax = binning["id"].max()
        result = binning[binning["id"] == idMax]
    if retWoe:
        return result.iloc[0].at['woe']
    else:
        count = len(binning)
        return result.iloc[0].at['id'] / (count - 1)
    

def lossFunction(y_pred, y_test):
    return np.mean(np.abs(y_pred - y_test) / y_test)


def getModelAuc(clf, x, y, is_plot=False):
    y_pred = clf.predict(x)
    y_score = clf.decision_function(x)
    fpr, tpr, thresholds = roc_curve(y, y_score)
    auc_score = auc(fpr, tpr)
    success_ratio = accuracy_score(y, y_pred)
    if is_plot:
        tuPlot.plotROC(y_score, y)
        tuPlot.plotKS(y_score, y, 10, 0)
    return auc_score, success_ratio


def validateSeqSelect(X, y, fs, clf, lf):
    x = X[fs]
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    score = 0
    for train, test in kf.split(x):
        clf.fit(x.iloc[train], y.iloc[train])
        train_auc, train_ratio = getModelAuc(clf, x.iloc[train], y.iloc[train])
        test_auc, test_ratio = getModelAuc(clf, x.iloc[test], y.iloc[test])
        score += (test_auc + train_auc) / 2
    return score / 5 - len(fs) * 0.005


def validateModel(clf, X, y, setType):
    y_pred = clf.predict(X)
    y_score = clf.predict_proba(X)[:, 1]
    success_ratio = accuracy_score(y, y_pred)
    print("%s set accuracy = %s" % (setType, success_ratio))
    ksDf = tuPlot.ksDf(y_score, y, 10)
    ksDf["set_type"] = setType
    print("%s set ks = %s" % (setType, ksDf))
    tuPlot.ksCurve(ksDf, 10, setType)
    tuPlot.plotROC(y_score, y, setType + "_roc.jpg")
    return ksDf