from tools.mysqlClient import getDfFromSql, saveDf, executeSql
from params import modBinning, modInputValue, modMapValue, modSelectedVar, modParam, modKs
from tools.utilsMonotonicWoeBinning import MonotonicWoeBinning
from tools.utilsSequenceSelection import Select
from tools.utilsGenerals import *
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import joblib


class FeatureEngineering:
    def __init__(self, db, keyColumns, labelCol, labelTable, varTables, dropColumns=None, qcutNum=1000, ivThd=0.02):
        self.db = db
        self.keyColumns = keyColumns  # 主键列
        self.labelCol = labelCol  # 标签列
        self.labelTable = labelTable  # 标签表
        self.varTables = varTables  # 变量表
        self.tableDict = getTableDict(varTables, labelTable)  # 表编号 y:标签表 t1~tn:变量表
        if dropColumns:
            self.dropColumns = dropColumns + keyColumns + [labelCol]
        else:
            self.dropColumns = keyColumns + [labelCol]
        self.qcutNum = qcutNum  # 预分组组数
        self.ivThd = ivThd  # 入模iv门限
        self.binInfo = pd.DataFrame()
        self.mappedDf = pd.DataFrame()
        self.modelDf = pd.DataFrame()

    # auto binning
    def woeBinning(self):
        executeSql(self.db, "truncate " + modBinning)
        mwb = MonotonicWoeBinning(y=self.labelCol, qcutNum=self.qcutNum)
        for tn in self.varTables:
            t = self.tableDict[tn]
            features = getDfFromSql(self.db, "describe " + tn)["Field"].tolist()
            features = [feature for feature in features if feature not in self.dropColumns]
            for feature in features:
                start = datetime.today()
                sql = "select y.{lb}, {t}.{f} from {lbt} y left join {tn} {t} on ".format(
                    lb=self.labelCol, lbt=self.labelTable, tn=tn, t=t, f=feature) + " and ".join(
                    ["y.{kc} = {t}.{kc}".format(kc=kc, t=t) for kc in self.keyColumns])
                featureDf = getDfFromSql(self.db, sql)
                featureDf = checkColType(featureDf, feature)
                if not featureDf.empty:
                    mwb.fit(featureDf)
                    summaryDf = mwb.woe_summary
                    summaryDf["variable_table"] = tn
                    summaryDf["feature_name_uni"] = t + "_" + feature
                    saveDf(summaryDf, self.db, modBinning, True)
                    end = datetime.today()
                    print("%s.%s: IV=%.3f 与目标变量%s 计算耗时%s秒"
                          % (tn, feature, mwb.total_iv, "正相关" if mwb.sign else "负相关", (end - start).seconds))

    # map woe
    def mapValues(self):
        executeSql(self.db, "drop table if exists " + modMapValue)
        df = getDfFromSql(self.db, """select variable_table, feature_name_uni, feature_name, id, woe, labels from %s
        where iv_total > %s order by feature_name, id""" % (modBinning, self.ivThd))
        self.binInfo = pd.DataFrame(data=[r[:-1] + labels2Bound(r[-1]) for r in df.values.tolist()],
                                    columns=df.columns.tolist()[:-1] + ["inputvalue", "inputvalue_shift"])
        df = self.concatFeatures()
        self.mappedDf = df[self.keyColumns + [self.labelCol]]
        for feature in df.columns:
            if feature in self.dropColumns:
                continue
            featureBinInfo = self.binInfo[self.binInfo["feature_name_uni"] == feature]
            if featureBinInfo.empty:
                continue
            temp = df[feature]
            temp = temp.map(lambda x: mapScore(x, featureBinInfo, retWoe=True))  # todo: woe or score???
            self.mappedDf[feature] = temp
            print("%s映射完成" % feature)
        saveDf(self.mappedDf, self.db, modMapValue)

    def concatFeatures(self):
        sqlInfo = self.binInfo[["variable_table", "feature_name", "feature_name_uni"]].drop_duplicates()
        print("按门限%s过滤得到%s个变量" % (self.ivThd, len(sqlInfo)))
        sqlInfo = groupByKey(sqlInfo, "variable_table")
        sqlInfo["variable_table_uni"] = sqlInfo["variable_table"].apply(lambda x: self.tableDict[x])
        sqlInfo = sqlInfo[["variable_table", "variable_table_uni", "values"]].values.tolist()
    
        sqlCol = ["select y.*"]
        for r in sqlInfo:
            t = r[1]
            for f in r[2]:
                sqlCol.append("{t}.{f0} as {f1}".format(t=t, f0=f[0], f1=f[1]))
    
        sqlTable = " from %s y" % self.labelTable
        for r in sqlInfo:
            tn, t = r[0], r[1]
            sqlTable += " left join %s %s on " % (tn, t) + " and ".join(["y.{kc} = {t}.{kc}".format(kc=kc, t=t) for kc in self.keyColumns])
    
        sql = ", ".join(sqlCol) + sqlTable
        df = getDfFromSql(self.db, sql)
        saveDf(df, self.db, modInputValue)
        return df
        
    # sequence selection
    def seqSelect(self, readDf=False):
        if readDf:
            self.mappedDf = getDfFromSql(self.db, "select * from " + modMapValue)
        df = self.mappedDf.copy()
        print("总样本%s个，目标变量=1数量%s个" % (len(df), len(df[df[self.labelCol] == 1])))
        sf = Select(sequencePro=True, randomPro=True, crossPro=False, logfile='record.log')
        sf.importDf(df, label=self.labelCol)
        sf.importLossFunction(lossFunction, direction='ascend')
        sf.initialNonTrainableFeatures(self.dropColumns)
        sf.initialFeatures([])
        sf.generateCol()
        sf.clf = LogisticRegression()
        features = sf.run(validateSeqSelect)
        print("选中变量：", features)
        self.modelDf = df[self.keyColumns + [self.labelCol] + features]
        saveDf(self.modelDf, self.db, modSelectedVar)
        
    # modeling
    def trainAndTest(self, readDf=False):
        if readDf:
            self.modelDf = getDfFromSql(self.db, "select * from " + modSelectedVar)
        df = self.modelDf.copy()
        trainDf, testDf = train_test_split(df, test_size=0.1)
        trainY = trainDf[self.labelCol]
        trainX = trainDf.drop(columns=self.dropColumns)
        testY = testDf[self.labelCol]
        testX = testDf.drop(columns=self.dropColumns)
        clf = LogisticRegressionCV(cv=10, random_state=0).fit(trainX, trainY)
        trainKsDf = validateModel(clf, trainX, trainY, "train")
        testKsDf = validateModel(clf, testX, testY, "test")
        ksDf = trainKsDf.append(testKsDf)
        joblib.dump(clf, 'model.joblib')
        coefDf = pd.DataFrame()
        coefDf['feature'] = trainX.columns
        coefDf['coef'] = clf.coef_[0]
        print(coefDf)
        saveDf(coefDf, self.db, modParam)
        saveDf(ksDf, self.db, modKs)
        
        
if __name__ == "__main__":
    fe = FeatureEngineering(db="trade",
                            keyColumns=["code", "trade_date"],
                            labelCol="label",
                            labelTable="label_table",
                            varTables=["var_table_ta", "var_table_jq"])
    # fe.woeBinning()
    # fe.mapValues()
    # fe.seqSelect(True)
    fe.trainAndTest(True)