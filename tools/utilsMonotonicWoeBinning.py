import os
import pandas as pd
import scipy.stats as stats
import warnings
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from tools.mysqlClient import getDfFromSql

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 130)
warnings.filterwarnings("ignore")
os.getcwd()


class MonotonicWoeBinning(BaseEstimator, TransformerMixin):
    def __init__(self, y, qcutNum=-1, n_threshold=50, y_threshold=10, p_threshold=0.35):
        self.n_threshold = n_threshold
        self.y_threshold = y_threshold
        self.p_threshold = p_threshold
        self.y = y
        self.column = ""
        self.init_summary = pd.DataFrame()
        self.bin_summary = pd.DataFrame()
        self.pvalue_summary = pd.DataFrame()
        self.dataset = pd.DataFrame()
        self.woe_summary = pd.DataFrame()
        self.total_iv = 0.
        self.sign = False
        self.bins = object
        self.bucket = object
        self.qcutNum = qcutNum
        self.binningDic = {}

    def numEqualBinning(self, dataset, col):
        binning_col = col + '_bin'
        quantile = np.array([float(i) / self.qcutNum for i in range(self.qcutNum + 1)])  # qcutNum组，qcutNum + 1个分位点
        dataset[binning_col] = dataset[col].rank(pct=True).apply(lambda x: (quantile >= x).argmax())
        binningSum = pd.DataFrame()
        group_by_binning = dataset.groupby([binning_col], as_index=True)
        binningSum["min_bin"] = group_by_binning[col].min()  # todo: 是否需要把min改为上一级的max，以便分组连续？
        binningSum['max_bin'] = group_by_binning[col].max()
        binningSum.reset_index(inplace=True)
        return dataset, binningSum
    
    def generate_summary(self, sign):
        self.init_summary = self.dataset.groupby([self.column]).agg({self.y: {"means": "mean", "nsamples": "size", "std_dev": "std"}})
        self.init_summary.columns = self.init_summary.columns.droplevel(level=0)
        self.init_summary = self.init_summary[["means", "nsamples", "std_dev"]]
        self.init_summary = self.init_summary.reset_index()
        self.init_summary["del_flag"] = 0
        self.init_summary["std_dev"] = self.init_summary["std_dev"].fillna(0)
        self.init_summary = self.init_summary.sort_values([self.column], ascending=sign)
    
    def combine_bins(self):
        summary = self.init_summary.copy()
        while True:
            i = 0
            summary = summary[summary.del_flag != 1]
            summary = summary.reset_index(drop=True)
            while True:
                j = i + 1
                if j >= len(summary):
                    break
                if summary.iloc[j].means < summary.iloc[i].means:
                    i = i + 1
                    continue
                else:
                    while True:
                        n = summary.iloc[j].nsamples + summary.iloc[i].nsamples
                        m = (summary.iloc[j].nsamples * summary.iloc[j].means + summary.iloc[i].nsamples * summary.iloc[i].means) / n
                        if n == 2:
                            s = np.std([summary.iloc[j].means, summary.iloc[i].means])
                        else:
                            s = np.sqrt((summary.iloc[j].nsamples * (summary.iloc[j].std_dev ** 2) + summary.iloc[i].nsamples * (summary.iloc[i].std_dev ** 2)) / n)
                        summary.loc[i, "nsamples"] = n
                        summary.loc[i, "means"] = m
                        summary.loc[i, "std_dev"] = s
                        summary.loc[j, "del_flag"] = 1
                        j = j + 1
                        if j >= len(summary):
                            break
                        if summary.loc[j, "means"] < summary.loc[i, "means"]:
                            i = j
                            break
                if j >= len(summary):
                    break
            dels = np.sum(summary["del_flag"])
            if dels == 0:
                break
        self.bin_summary = summary.copy()
    
    def calculate_pvalues(self):
        summary = self.bin_summary.copy()
        while True:
            summary["means_lead"] = summary["means"].shift(-1)
            summary["nsamples_lead"] = summary["nsamples"].shift(-1)
            summary["std_dev_lead"] = summary["std_dev"].shift(-1)
            summary["est_nsamples"] = summary["nsamples_lead"] + summary["nsamples"]
            summary["est_means"] = (summary["means_lead"] * summary["nsamples_lead"] + summary["means"] * summary["nsamples"]) / summary["est_nsamples"]
            summary["est_std_dev2"] = (summary["nsamples_lead"] * summary["std_dev_lead"] ** 2 + summary["nsamples"] * summary["std_dev"] ** 2) / (summary["est_nsamples"] - 2)
            summary["z_value"] = (summary["means"] - summary["means_lead"]) / np.sqrt(summary["est_std_dev2"] * (1 / summary["nsamples"] + 1 / summary["nsamples_lead"]))
            summary["p_value"] = 1 - stats.norm.cdf(summary["z_value"])
            summary["p_value"] = summary.apply(
                lambda row: row["p_value"] + 1
                if (row["nsamples"] < self.n_threshold)\
                or (row["nsamples_lead"] < self.n_threshold)\
                or (row["means"] * row["nsamples"] < self.y_threshold)\
                or (row["means_lead"] * row["nsamples_lead"] < self.y_threshold) else row["p_value"], axis=1)
            max_p = max(summary["p_value"])
            row_of_maxp = summary['p_value'].idxmax()
            row_delete = row_of_maxp + 1
            if max_p > self.p_threshold:
                summary = summary.drop(summary.index[row_delete])
                summary = summary.reset_index(drop=True)
            else:
                break
            summary["means"] = summary.apply(
                lambda row: row["est_means"] if row["p_value"] == max_p else row["means"], axis=1)
            summary["nsamples"] = summary.apply(
                lambda row: row["est_nsamples"] if row["p_value"] == max_p else row["nsamples"], axis=1)
            summary["std_dev"] = summary.apply(
                lambda row: np.sqrt(row["est_std_dev2"]) if row["p_value"] == max_p else row["std_dev"], axis=1)
        self.pvalue_summary = summary.copy()
    
    def calculate_woe(self):
        woe_summary = self.pvalue_summary[[self.column, "nsamples", "means"]]
        woe_summary["bads"] = woe_summary["means"] * woe_summary["nsamples"]
        woe_summary["goods"] = woe_summary["nsamples"] - woe_summary["bads"]
        total_goods = np.sum(woe_summary["goods"])
        total_bads = np.sum(woe_summary["bads"])
        woe_summary["dist_good"] = woe_summary["goods"] / total_goods
        woe_summary["dist_bad"] = woe_summary["bads"] / total_bads
        woe_summary["WOE_" + self.column] = np.log(woe_summary["dist_good"] / woe_summary["dist_bad"])
        woe_summary["iv_components"] = (woe_summary["dist_good"] - woe_summary["dist_bad"]) * woe_summary["WOE_" + self.column]
        total_iv = np.sum(woe_summary["iv_components"])
        return woe_summary, total_iv
    
    def doubleCheck(self):
        self.generate_summary(True)
        self.combine_bins()
        self.calculate_pvalues()
        woe_summary_true, total_iv_true = self.calculate_woe()

        self.generate_summary(False)
        self.combine_bins()
        self.calculate_pvalues()
        woe_summary_false, total_iv_false = self.calculate_woe()
        
        if total_iv_true > total_iv_false:
            self.woe_summary = woe_summary_true
            self.total_iv = total_iv_true
            self.sign = True
        else:
            self.woe_summary = woe_summary_false
            self.total_iv = total_iv_false
            self.sign = False
    
    def generate_bin_labels(self, row):
        return " ~ ".join(map(str, np.sort([row[self.column], row[self.column + "_shift"]])))
    
    def generate_final_dataset(self):
        if self.sign is False:
            shift_var = -1
            # shift_var = 1
            self.bucket = True
        else:
            shift_var = -1
            self.bucket = False
        self.woe_summary[self.column + "_shift"] = self.woe_summary[self.column].shift(shift_var)
        if self.sign is False:
            # self.woe_summary.loc[0, self.column + "_shift"] = -np.inf
            self.woe_summary.loc[len(self.woe_summary) - 1, self.column + "_shift"] = -np.inf
            self.bins = np.sort(list(self.woe_summary[self.column]) + [np.Inf, -np.Inf])
        else:
            self.woe_summary.loc[len(self.woe_summary) - 1, self.column + "_shift"] = np.inf
            self.bins = np.sort(list(self.woe_summary[self.column]) + [np.Inf, -np.Inf])
        self.woe_summary["labels"] = self.woe_summary.apply(self.generate_bin_labels, axis=1)
        self.dataset["bins"] = pd.cut(self.dataset[self.column], self.bins, right=self.bucket, precision=1)
        self.dataset["bins"] = self.dataset["bins"].astype(str)
        self.dataset['bins'] = self.dataset['bins'].map(lambda x: x.lstrip('[').rstrip(')'))

    def getBinningMin(self, binningNum):
        if np.isinf(binningNum):
            return float(binningNum)
        return self.binningDic[self.binningDic[self.column] == binningNum]['min_bin'].iloc[0]

    def getBinningMax(self, binningNum):
        if np.isinf(binningNum):
            return float(binningNum)
        return self.binningDic[self.binningDic[self.column] == binningNum]['max_bin'].iloc[0]
    
    def convertWoeSummary(self, isTransfer):
        # self.woe_summary.rename(columns={self.column: 'left_value',
        #                                  self.column + "_shift": 'right_value',
        #                                  'WOE_' + self.column: 'woe'}, inplace=True)
        if self.sign:
            self.woe_summary.rename(columns={self.column: 'left_value',
                                             self.column + "_shift": 'right_value',
                                             'WOE_' + self.column: 'woe'}, inplace=True)
        else:
            self.woe_summary.rename(columns={self.column: 'right_value',
                                             self.column + "_shift": 'left_value',
                                             'WOE_' + self.column: 'woe'}, inplace=True)

        if isTransfer:
            self.woe_summary['left_value'] = self.woe_summary['left_value'].apply(lambda x: self.getBinningMin(x))
            self.woe_summary['right_value'] = self.woe_summary['right_value'].apply(lambda x: self.getBinningMax(x))
            self.woe_summary['labels'] = self.woe_summary.apply(lambda x: '%s ~ %s' % (x['left_value'], x['right_value']), axis=1)
            self.column = self.column[:-4]
            
        self.woe_summary["feature_name"] = self.column
        self.woe_summary["iv_total"] = self.total_iv
        self.woe_summary = self.woe_summary.reset_index()
        self.woe_summary = self.woe_summary.rename(columns={"index": "id"})
        columns = ['feature_name', 'id', 'labels', 'woe', 'iv_components', 'iv_total', 'nsamples', 'means', 'bads',
                   'goods', 'dist_good', 'dist_bad']
        self.woe_summary = self.woe_summary[columns]

    def fit(self, dataset):
        cleanDataSet = dataset.dropna()
        self.column = cleanDataSet.columns[cleanDataSet.columns != self.y][0]
        valueCount = cleanDataSet[self.column].value_counts().count()
        isTransfer = False
        if valueCount > self.qcutNum != -1:
            cleanDataSet, self.binningDic = self.numEqualBinning(cleanDataSet, self.column)
            self.column = self.column + "_bin"
            self.dataset = cleanDataSet[[self.column, self.y]]
            isTransfer = True
        else:
            self.dataset = cleanDataSet
        self.doubleCheck()
        self.generate_final_dataset()
        self.convertWoeSummary(isTransfer)
        
        
if __name__ == "__main__":
    mwb = MonotonicWoeBinning(y="label", qcutNum=1000)
    feature = "ta_mom_5_d"
    keyColumns = ["code", "trade_date"]
    sql = "select y.{lb}, {t}.{f} from {lbt} y left join {tn} {t} on ".format(
        lb="label", lbt="label_table", tn="var_table_ta", t="v1", f=feature) + " and ".join(
        ["y.{kc} = {t}.{kc}".format(kc=kc, t="v1") for kc in keyColumns])
    print(sql)
    featureDf = getDfFromSql("trade", sql)
    mwb.fit(featureDf)