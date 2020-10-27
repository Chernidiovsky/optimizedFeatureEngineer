# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
import scipy.stats.stats as stats
import scipy.stats as stat
import matplotlib.pyplot as plt
from sklearn.linear_model import RandomizedLogisticRegression as RLR
from sklearn.linear_model import RandomizedLasso as RLS
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2, f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pendulum as pen


def cal_bin(x, n):
    the_pct = np.linspace(0, 100, n)  # 等分非缺失样本为20份
    sbins = np.percentile(x, the_pct)
    auto_bins = np.unique(sbins)  # 排除重复值影响，众数可能会造成此现象
    return auto_bins


def map_bin(x, bins):
    kwargs = {}
    if x == max(bins):
        kwargs['right'] = True
    the_bin = bins[np.digitize([x], bins, **kwargs)[0]]
    bin_lower = bins[np.digitize([x], bins, **kwargs)[0] - 1]
    return '{0}~{1}'.format(round(bin_lower, 4), round(the_bin, 4))


class FeatureBinning(object):
    def __init__(self, bin_type=2, criterion='entropy', max_leaf_nodes=8, min_samples_leaf=100, max_depth=4,
                 bin_count=20, n_thresholdRatio=0.05, defaults_threshold=30, p_threshold=0.05, na=-999, bins_dict=None):
        """
        bin_type:{1:等分分箱，2：决策数分箱,3,线性拟合分箱，4:分类变量进行分箱合并，5:分类变量不分箱,6：手动分箱, 7:单调最优分箱}
        """
        self.na = na
        self.bin_type = bin_type
        self.criterion = criterion
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.bin_count = bin_count
        self.bins_dict = bins_dict
        self.n_thresholdRatio = n_thresholdRatio
        self.defaults_threshold = defaults_threshold
        self.p_threshold = p_threshold
    
    def equal_split(self, var, y, bin_count=None):
        pat = r"\(|\)|\[|\]"
        try:
            bucket = pd.qcut(var, bin_count, duplicates='drop').astype(str)
            bucket = bucket.str.replace(pat, "").str.replace(', ', '~')
        except:
            bins = cal_bin(var, bin_count)
            if var.count() != len(var):
                bins = np.concatenate(([self.na], bins))
                bucket = var.apply(map_bin, bins=bins)
            else:
                bucket = var.apply(map_bin, bins=bins)
        return bucket
    
    def decision_split(self, var, y):
        min_samples_leaf = min(len(y) / 10, self.min_samples_leaf)
        dt = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth,
                                    max_leaf_nodes=self.max_leaf_nodes, min_samples_leaf=min_samples_leaf)
        var2 = var.values.reshape(-1, 1)
        dt.fit(var2, y)
        bins = np.unique(dt.tree_.threshold[dt.tree_.feature > -2])
        if len(bins) > 0:
            bins[0] = -np.inf
            bins[-1] = np.inf
            bins = np.insert(bins, 1, -998)
            var = var.fillna(-999)
            bucket = var.apply(map_bin, bins=bins)
        else:
            bucket = None
        return bucket
    
    def line_split(self, var, y, bin_count=None):
        r = 0
        n = bin_count
        bucket = None
        while np.abs(r) < 1 and n >= 5:
            bins = cal_bin(var.dropna(), n)
            if var.count() != len(var):
                bins = np.concatenate(([self.na], bins))
                bucket = var.apply(map_bin, bins=bins)
            else:
                bucket = var.apply(map_bin, bins=bins)
                d1 = pd.DataFrame({"variable": var, "Y": y, "Bucket": bucket})
                d2 = d1.groupby('Bucket', as_index=False)
                r, p = stats.spearmanr(d2.mean().variable, d2.mean().Y)
            n = n - 1
        return bucket
    
    def cate_split(self, var, y):
        bucket = None
        if self.bins_dict is None:
            print('Please set bins_dict')
        else:
            if var.name in self.bins_dict:
                bins = self.bins_dict[var.name]
                var_unique = var.unique().tolist()
                labels_dict = {}
                for i in bins:
                    for j in var_unique:
                        if j in i:
                            labels_dict[j] = ','.join(i)
                bucket = var.map(labels_dict)
        return bucket
    
    def no_split(self, var):
        bucket = var
        return bucket
    
    def hand_split(self, var, y, sign=False):
        if self.bins_dict is None:
            print('Please set bins_dict')
            bucket = None
        else:
            if var.name in self.bins_dict:
                bins = self.bins_dict[var.name]
                labels = ["{0}~{1}".format(bins[j], bins[j + 1]) for j in range(0, len(bins) - 1)]
                bucket = pd.cut(var, bins=bins, include_lowest=True, labels=labels).astype(str)
            else:
                print(str(var.name) + ' is not in bins_dict')
                bucket = None
        return bucket
    
    def monotone_optimal_split(self, var, y, sign=False):
        inputMax = var.max()
        if np.abs(inputMax) >= 1000.0:
            scaler = np.power(10, int(np.log10(inputMax)))
        else:
            scaler = 1.0
        var = (var / scaler).round(5)
        n_threshold = int(var.count() * self.n_thresholdRatio)
        div_ratio = 1.0
        df = pd.DataFrame(dict(var=var, y=y))
        df = df.loc[pd.notnull(df['var']), :]
        df = df.apply(lambda x: x.astype(float), axis=0)
        df_group = df.groupby(var).agg({'y': ['mean', 'size', 'std']}, drop_index=True)
        df_group.columns = ["means", "nsamples", "std_dev"]
        df_group = df_group.reset_index()
        df_group['del_flag'] = 0
        df_group["std_dev"] = df_group["std_dev"].fillna(0)
        df_group = df_group.sort_values(var.name, ascending=sign)
        while True:
            i = 0
            df_group = df_group[df_group.del_flag != 1]
            df_group = df_group.reset_index(drop=True)
            while True:
                j = i + 1
                if j >= len(df_group):
                    break
                if df_group.iloc[j].means < df_group.iloc[i].means:
                    i = i + 1
                    continue
                else:
                    while True:
                        n = df_group.iloc[j].nsamples + df_group.iloc[i].nsamples
                        m = (df_group.iloc[j].nsamples * df_group.iloc[j].means + df_group.iloc[i].nsamples *
                             df_group.iloc[i].means) / n
                        if n == 2:
                            s = np.std([df_group.iloc[j].means, df_group.iloc[i].means])
                        else:
                            s = np.sqrt((df_group.iloc[j].nsamples * (df_group.iloc[j].std_dev ** 2) + df_group.iloc[
                                i].nsamples * (df_group.iloc[i].std_dev ** 2)) / n)
                        df_group.loc[i, "nsamples"] = n
                        df_group.loc[i, "means"] = m
                        df_group.loc[i, "std_dev"] = s
                        df_group.loc[j, "del_flag"] = 1
                        j = j + 1
                        if j >= len(df_group):
                            break
                        if df_group.loc[j, "means"] < df_group.loc[i, "means"]:
                            i = j
                            break
                if j >= len(df_group):
                    break
            dels = np.sum(df_group["del_flag"])
            if dels == 0:
                break
        while True:
            df_group["means_lead"] = df_group["means"].shift(-1)
            df_group["nsamples_lead"] = df_group["nsamples"].shift(-1)
            df_group["std_dev_lead"] = df_group["std_dev"].shift(-1)
            df_group["est_nsamples"] = df_group["nsamples_lead"] + df_group["nsamples"]
            df_group["est_means"] = (df_group["means_lead"] * df_group["nsamples_lead"] + df_group["means"] * df_group[
                "nsamples"]) / df_group["est_nsamples"]
            df_group["est_std_dev2"] = (df_group["nsamples_lead"] * df_group["std_dev_lead"] ** 2 + df_group[
                "nsamples"] * df_group["std_dev"] ** 2) / (df_group["est_nsamples"] - 2)
            df_group["z_value"] = (df_group["means"] - df_group["means_lead"]) / np.sqrt(
                df_group["est_std_dev2"] * (1 / df_group["nsamples"] + 1 / df_group["nsamples_lead"]))
            df_group["p_value"] = 1 - stat.norm.cdf(df_group["z_value"])
            condition = (df_group["nsamples"] < n_threshold) | (df_group["nsamples_lead"] < n_threshold) | (
                        df_group["means"] * df_group["nsamples"] < self.defaults_threshold) | (
                                    df_group["means_lead"] * df_group["nsamples_lead"] < self.defaults_threshold)
            df_group[condition].p_value = df_group[condition].p_value + 1
            df_group["p_value"] = df_group.apply(lambda row: row["p_value"] + 1 if (row["nsamples"] < n_threshold) | (
                        row["nsamples_lead"] < n_threshold) | (row["means"] * row[
                "nsamples"] < self.defaults_threshold) | (row["means_lead"] * row[
                "nsamples_lead"] < self.defaults_threshold) else row["p_value"], axis=1)
            max_p = max(df_group["p_value"])
            row_of_maxp = df_group["p_value"].idxmax()
            row_delete = row_of_maxp + 1
            if max_p > self.p_threshold:
                df_group = df_group.drop(df_group.index[row_delete])
                df_group = df_group.reset_index(drop=True)
            else:
                break
            df_group["means"] = df_group.apply(
                lambda row: row["est_means"] if row["p_value"] == max_p else row["means"], axis=1)
            df_group["nsamples"] = df_group.apply(
                lambda row: row["est_nsamples"] if row["p_value"] == max_p else row["nsamples"], axis=1)
            df_group["std_dev"] = df_group.apply(
                lambda row: np.sqrt(row["est_std_dev2"]) if row["p_value"] == max_p else row["std_dev"], axis=1)
        bins = (np.unique(df_group[var.name]).astype(float) - 1e-4) * scaler
        if len(bins) > 0:
            bins[0] = -np.inf
            bins = np.insert(bins, 1, -998)
            bins[-1] = np.inf
            var = var * scaler
            var = var.fillna(-999)
            bucket = var.apply(map_bin, bins=bins)
        else:
            bucket = None
        crosstab = pd.crosstab(bucket, y)
        crosstab = crosstab.rename(columns={0: 'good_count', 1: 'bad_count', '0': 'good_count', '1': 'bad_count'})
        crosstab['good_rate'] = crosstab['good_count'] / crosstab['good_count'].sum()
        crosstab['bad_rate'] = crosstab['bad_count'] / crosstab['bad_count'].sum()
        crosstab['woe'] = np.log(crosstab['good_rate'] / crosstab['bad_rate']).replace([np.inf, -np.inf], np.nan)
        crosstab['iv'] = (crosstab['good_rate'] - crosstab['bad_rate']) * crosstab['woe']
        IV = crosstab['iv'].sum()
        return bucket, IV
    
    def binning_series(self, var, y, bin_type=None, bin_count=None):
        bin_type = bin_type if bin_type in [1, 2, 3, 4, 5, 6, 7] else self.bin_type
        bin_count = bin_count if bin_count else self.bin_count
        var1 = var.fillna(self.na)
        if bin_type == 1:
            bucket = self.equal_split(var1, y, bin_count=bin_count)
        elif bin_type == 2:
            bucket = self.decision_split(var1, y)
            if bucket is None:
                bucket = self.no_split(var1)
            else:
                bucket
        elif bin_type == 3:
            bucket = self.line_split(var1, y, bin_count=bin_count)
        elif bin_type == 4:
            bucket = self.cate_split(var1, y)
        elif bin_type == 5:
            bucket = self.no_split(var1)
        elif bin_type == 6:
            bucket = self.hand_split(var1, y)
        elif bin_type == 7:
            bucket_ascend, iv_ascend = self.monotone_optimal_split(var, y, sign=True)
            bucket_descend, iv_descend = self.monotone_optimal_split(var, y, sign=False)
            if iv_ascend < iv_descend:
                bucket = bucket_descend
            else:
                bucket = bucket_ascend
        else:
            print(u'已没有其他选项')
        return bucket
    
    def binning_df(self, df, binning_feature, label, bin_type=None, bin_count=None):
        df_columns = df.columns.tolist()
        df_columns.remove(label)
        in_columns = [i for i in df_columns if i in binning_feature]
        out_columns = [i for i in binning_feature if i not in df_columns]
        if len(out_columns) > 0:
            print(out_columns, 'not in dataframe!')
        for i in in_columns:
            bucket = self.binning_series(df[i], df[label], bin_type=bin_type, bin_count=bin_count)
            df[i + '_bin'] = bucket
        return df


class FeatureStats(object):
    def __init__(self, binning_var, y, bad_weight=1, good_weight=1, trans_df=pd.DataFrame(), datatype='num',
                 qushi='up'):
        self.binning_var = binning_var
        self.y = y
        self.datatype = datatype
        self.qushi = qushi
        self.trans_df = pd.crosstab(binning_var, y)
        self.trans_df = self.trans_df.rename(
            columns={0: 'good_count', 1: 'bad_count', '0': 'good_count', '1': 'bad_count'})
        good_bad = ['good_count', 'bad_count']
        for i in good_bad:
            if i not in self.trans_df.columns:
                self.trans_df[i] = 0
        self.trans_df['total'] = self.trans_df.sum(axis=1)
        self.trans_df['good_count_weight'] = self.trans_df['good_count'] * good_weight
        self.trans_df['bad_count_weight'] = self.trans_df['bad_count'] * bad_weight
        self.trans_df['total_weight'] = self.trans_df['good_count_weight'] + self.trans_df['bad_count_weight']
        self.good_total = self.trans_df['good_count_weight'].sum()
        self.bad_total = self.trans_df['bad_count_weight'].sum()
        self.all = self.good_total + self.bad_total
        self.trans_df['bin_pct'] = self.trans_df['total_weight'] / self.all
        self.trans_df['bad_rate'] = self.trans_df['bad_count_weight'].div(self.trans_df['total_weight'])
        self.trans_df['sample_bad_rate'] = self.bad_total / self.all
        self.good_dist = np.nan_to_num(self.trans_df['good_count_weight'] / self.good_total)
        self.bad_dist = np.nan_to_num(self.trans_df['bad_count_weight'] / self.bad_total)
        if not (self.good_dist / self.bad_dist).all():
            self.trans_df['woe'] = np.nan_to_num(np.log(self.good_dist / self.bad_dist))
        else:
            self.trans_df['woe'] = np.log(self.good_dist / self.bad_dist)
        self.trans_df['woe'] = round(self.trans_df['woe'], 4)
        self.trans_df['iv_i'] = ((self.good_dist - self.bad_dist) * self.trans_df['woe']).replace([np.inf, -np.inf],
                                                                                                  np.nan)
        self.trans_df['iv'] = self.trans_df['iv_i'].sum()
        self.trans_df['iv'] = round(self.trans_df['iv'], 4)
        # self.trans_df = self.trans_df.reset_index()
        if self.datatype == 'num':
            self.trans_df['min_value'] = self.trans_df.index.map(
                lambda x: float(x.split('~')[0]) if isinstance(x, str) else x)
        else:
            self.trans_df['min_value'] = self.trans_df['woe']
        self.woe_df = self.trans_df.sort_values(by='min_value')\
            .drop(['iv_i', 'min_value'], axis=1) if self.qushi == 'up' else self.trans_df.sort_values(
            by='min_value', ascending=False).drop(['iv_i', 'min_value'], axis=1)
        self.woe_df['bad_cum'] = (self.woe_df.bad_count_weight / self.woe_df.bad_count_weight.sum()).cumsum()
        self.woe_df['good_cum'] = (self.woe_df.good_count_weight / self.woe_df.good_count_weight.sum()).cumsum()
        self.woe_df = self.woe_df.drop(['good_count_weight', 'bad_count_weight', 'total_weight'], axis=1)
    
    def cal_gini(self):
        woe_df = self.trans_df.sort_values(by='woe').drop('iv_i', axis=1)
        woe_df['bad_cum'] = (woe_df.bad_count_weight / woe_df.bad_count_weight.sum()).cumsum()
        woe_df['good_cum'] = (woe_df.good_count_weight / woe_df.good_count_weight.sum()).cumsum()
        cumGood = np.concatenate([[0], woe_df['good_cum']])
        cumBad = np.concatenate([[0], woe_df['bad_cum']])
        area = 0
        for i in range(1, len(cumGood)):
            area += 1 / 2 * (cumBad[i - 1] + cumBad[i]) * (cumGood[i] - cumGood[i - 1])
        gini = 2 * (area - 0.5)
        return gini
    
    def cal_ks(self):
        ks = (self.woe_df['bad_cum'] - self.woe_df['good_cum']).round(3)
        max_ks = ks.apply(lambda x: abs(x) if abs(x) == ks.abs().max() else np.nan)
        return ks, max_ks
    
    def stats_feature(self):
        self.woe_df['gini'] = round(self.cal_gini(), 2)
        ks_result = self.cal_ks()
        self.woe_df['ks'] = ks_result[0]
        self.woe_df['max_ks'] = ks_result[1]
        var = self.woe_df.index.name
        woe_df = self.woe_df.reset_index().rename(columns={var: 'bin_group'})
        woe_df['var'] = var
        cols = woe_df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        woe_df = woe_df[cols]
        return woe_df


def feature_stats_series(df, one_feature, label, bins_dict=None, bin_count=20, bin_type=2, bad_weight=1, good_weight=1,
                         min_samples_leaf=2000, datatype='num', qushi='up'):
    """
    对单一变量手工分箱，默认使用分类变量不分箱选项，返回分箱统计
    df:输入DataFrame
    one_feature:要分箱的列名，str
    label:目标变量
    bins_dict：默认为None，采用手动分箱时使用
    bin_type：分箱类型，1:等分分箱，2：决策数分箱,3,线性拟合分箱，
                      4:分类变量进行分箱合并，5:分类变量不分箱,6：手动分箱，
                      默认是5
    min_samples_leaf：决策树分箱的最小叶节点样本数
    """
    b_df = df.copy()[[one_feature, label]]
    bin_df = FeatureBinning(min_samples_leaf=min_samples_leaf, bins_dict=bins_dict, bin_type=bin_type,
                            bin_count=bin_count).binning_df(b_df, [one_feature], label)
    feature_stats = FeatureStats(bin_df[one_feature + '_bin'], bin_df[label], bad_weight=bad_weight,
                                 good_weight=good_weight, datatype=datatype, qushi=qushi).stats_feature()
    return feature_stats


def feature_bin_hand(df, label, bins_dict=None, bad_weight=1, good_weight=1):
    """
    使用分好箱的bins_dict对Dataframe进行全量手工分箱,返回分箱
    df:输入DataFrame
    label:目标变量
    bins_dict：分好箱的字典
    """
    if bins_dict:
        model_columns = list(bins_dict.keys())
        bin_df = FeatureBinning(bins_dict=bins_dict).binning_df(df, model_columns, label, bin_type=6)
        return bin_df
    else:
        print('请提供手工分箱的字典')
        return {}


def feature_stats_all(df, all_feature, label, bad_weight=1, good_weight=1, datatype='num', qushi='up'):
    """
    对分箱变量进行统计
    df:输入分好箱的DataFrame
    all_feature:分好箱的所有列名的list
    label:目标变量
    """
    df_columns = df.columns.tolist()
    df_columns.remove(label)
    in_columns = [i for i in df_columns if i in all_feature]
    out_columns = [i for i in all_feature if i not in df_columns]
    if len(out_columns) > 0:
        print(out_columns, 'not in dataframe!')
    stats_all_df = pd.DataFrame()
    for i in in_columns:
        feature_stats = FeatureStats(df[i], df[label], bad_weight=bad_weight, good_weight=good_weight,
                                     datatype=datatype).stats_feature()
        stats_all_df = stats_all_df.append(feature_stats)
    return stats_all_df


def map_woe(bin_df, woe_map, model_bin_columns):
    """
    将分好箱的Dataframe进行WOE Map
    bin_df：使用feature_bin_hand方法分好箱的Dataframe
    woe_map: feature_stats_all统计好的分箱对应
    model_bin_columns: 分好箱的字段
    """
    model_stats_left = woe_map.loc[:, ['var', 'bin_group', 'woe']]
    woe_dict = {}
    for i in model_stats_left['var'].unique():
        temp = model_stats_left.loc[model_stats_left['var'] == i][['bin_group', 'woe']]
        temp1 = temp.set_index('bin_group')
        woe_dict[i] = temp1.to_dict()['woe']
    for i in model_bin_columns:
        bin_df[i + '_woe'] = bin_df[i].map(woe_dict[i])
    return bin_df


def auto_woe(df, all_feature, label, min_samples_leaf):
    """
    利用决策树分箱自动分箱，woe编码
    """
    bin_df = FeatureBinning(min_samples_leaf=min_samples_leaf).binning_df(df, all_feature, label, bin_type=2)
    bin_columns = [i for i in bin_df.columns if i.endswith('bin')]
    stats_df = feature_stats_all(bin_df, bin_columns, label)
    woe_df = map_woe(bin_df, stats_df, bin_columns)
    return woe_df, stats_df


def dt_auto_bin_select(df, all_feature, label, min_samples_leaf, max_leaf_nodes, iv_threshold=0.02):
    bin_df = FeatureBinning(min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes).binning_df(df,
                                                                                                         all_feature,
                                                                                                         label)
    bin_cols = [i + '_bin' for i in all_feature]
    stats_df = feature_stats_all(bin_df, bin_cols, label)
    stats_left = stats_df.loc[stats_df.iv >= iv_threshold]
    var_left = stats_left['var'].unique().tolist()
    feature_left = [i.split('_bin')[0] for i in var_left]
    return stats_left, feature_left


def mop_auto_bin_select(df, all_feature, label, n_thresholdRatio, defaults_threshold, p_threshold=0.05,
                        iv_threshold=0.02):
    bin_df = FeatureBinning(bin_type=7, n_thresholdRatio=n_thresholdRatio,
                            defaults_threshold=defaults_threshold).binning_df(df, all_feature, label, bin_type=7)
    bin_cols = [i + '_bin' for i in all_feature]
    stats_df = feature_stats_all(bin_df, bin_cols, label)
    stats_left = stats_df.loc[stats_df.iv >= iv_threshold]
    var_left = stats_left['var'].unique().tolist()
    feature_left = [i.split('_bin')[0] for i in var_left]
    return stats_left, feature_left


def auto_bin_select(df, all_feature, label, min_samples_leaf, max_leaf_nodes=8, dt_iv_threshold=0.02,
                    n_thresholdRatio=0.08, defaults_threshold=5, p_threshold=0.05, mop_iv_threshold=0.02):
    dt_stats_df, dt_feature_cols = dt_auto_bin_select(df, all_feature, label, min_samples_leaf=min_samples_leaf,
                                                      max_leaf_nodes=max_leaf_nodes, iv_threshold=dt_iv_threshold)
    mop_stats_df, mop_feature_cols = mop_auto_bin_select(df, dt_feature_cols, label, n_thresholdRatio=n_thresholdRatio,
                                                         defaults_threshold=defaults_threshold,
                                                         iv_threshold=mop_iv_threshold)
    return mop_stats_df, mop_feature_cols


def condition_stats(df, all_feature, label, condition_var, bins_dict=None, bad_weight=1, good_weight=1, datatype='num',
                    qushi='up'):
    bin_df = feature_bin_hand(df, label, bins_dict=bins_dict, bad_weight=1, good_weight=1)
    all_condition = bin_df[condition_var].unique().tolist()
    bin_var = [i + '_bin' for i in all_feature]
    condition_stats_list = []
    for i in all_condition:
        sub_bin_df = bin_df.loc[bin_df[condition_var] == i]
        sub_stats_df = feature_stats_all(sub_bin_df, bin_var, label, bad_weight=bad_weight, good_weight=good_weight,
                                         datatype=datatype, qushi=qushi)
        sub_stats_df['time_interval'] = i
        condition_stats_list.append(sub_stats_df)
    condition_stats_df = pd.concat(condition_stats_list, ignore_index=True)
    return condition_stats_df


"""
时间变量转换
"""


def pendulum_parse(time_variable, frequency):
    if frequency.upper() == 'D':
        time_str = pen.parse(time_variable).day
    elif frequency.upper() == 'W':
        time_str = pen.parse(time_variable).week
    elif frequency.upper() == 'M':
        time_str = pen.parse(time_variable).month
    else:
        time_str = time_variable
        print('unsupported frequency!')
    return time_str


def datetime_parse(time_variable, frequency):
    if frequency.upper() == 'D':
        time_str = time_variable.day
    elif frequency.upper() == 'W':
        time_str = time_variable.week
    elif frequency.upper() == 'M':
        time_str = time_variable.month
    else:
        time_str = time_variable
        print('unsupported frequency!')
    return time_str


def time_map(time_variable, frequency):
    if str(type(time_variable)) == "<class 'datetime.datetime'>" or str(
            type(time_variable)) == "<class 'datetime.date'>":
        time_str = datetime_parse(time_variable, frequency)
        return time_str
    elif str(type(time_variable)) == "<class 'str'>":
        try:
            time_str = pendulum_parse(time_variable, frequency)
            return time_str
        except:
            print('unsupported time_variable!')
    else:
        print('only support str or datetime')


def time_trans(df, time_variable, frequency):
    df = df.copy()
    df['time_interval'] = df[time_variable].apply(time_map, args=(frequency,))
    return df


def stable_select(df, y, rd_reg_columns, threshold=0.2, model='rlr'):
    X = df.loc[:, rd_reg_columns]
    Y = df[y]
    if model == 'rlr':
        rlr = RLR(scaling=0.5, sample_fraction=0.75, n_resampling=300, selection_threshold=threshold)  # 随机逻辑回归
        rlr.fit(X, Y)
        scores = rlr.scores_
    elif model == 'rls':
        rls = RLS(scaling=0.5, sample_fraction=0.75, n_resampling=300, selection_threshold=threshold)  # 随机Lasso回归
        rls.fit(X, Y)
        scores = rls.scores_
    elif model == 'rfr':
        rf = RFR()
        rf.fit(X, Y)
        scores = rf.feature_importances_
    else:
        pass
    result = pd.Series(dict(zip(X.columns, scores))).rename('score').sort_values(ascending=False)
    plt.figure(figsize=(20, 10))
    result.plot.barh(title='Feature Importances', color='lightblue')
    plt.ylabel('Feature Importance Score')
    return result


def f_select(df, y, sel_columns, model='f_classif'):
    X = df.loc[:, sel_columns]
    Y = df[y]
    if model == 'f_classif':
        selectk = SelectKBest(f_classif, k='all')
    elif model == 'f_regression':
        selectk = SelectKBest(f_regression, k='all')
    elif model == 'chi2':
        selectk = SelectKBest(chi2, k='all')
    else:
        pass
    selectk.fit(X, Y)
    scores = selectk.scores_
    result = pd.Series(dict(zip(X.columns, scores))).rename('f_score')
    return result


def xgb_select(df, y, sel_columns):
    X = df.loc[:, sel_columns]
    Y = df[y]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1301, stratify=Y, test_size=0.3)
    clf = xgb.XGBClassifier(max_depth=5, n_estimators=1000, learning_rate=0.02, objective='binary:logistic', nthread=4,
                            subsample=0.85, colsample_bytree=0.75, seed=4242)
    clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc", eval_set=[(X_test, y_test)])
    print('Overall AUC:', roc_auc_score(Y, clf.predict_proba(X)[:, 1]))
    # y_pred = clf.predict_proba(X_test)
    # mapFeat = dict(zip(["f"+str(i) for i in range(len(X.columns))],X.columns))
    ts = pd.Series(clf.get_booster().get_fscore())
    feature_list = [x for x in X_train]
    feat_imp = pd.Series(clf.feature_importances_, feature_list).sort_values(ascending=False)
    plt.figure(figsize=(20, 10))
    feat_imp.plot.barh(title='Feature Importances', color='lightblue')
    plt.ylabel('Feature Importance Score')
    return ts