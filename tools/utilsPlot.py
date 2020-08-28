import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

plt.style.use('ggplot')


def plotROC(preds, labels, name="roc.jpg"):
    y_pred = preds  # 预测值
    y_true = labels  # 真实值，bad为1，good为0
    fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label=1)  # 计算真正率和假正率
    roc_auc = auc(fpr, tpr)  # 计算auc的值
    plt.clf()
    plt.figure()
    lw = 2
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Roc Curve')
    plt.legend(loc="lower right")
    plt.savefig(name)


def ksDf(score, label, num=10, good=0, bad=1):
    # 1.将数据从小到大平均分成num组
    y_pre = 'score'
    y_true = 'label'
    df = pd.DataFrame({'score': score, 'label': label})
    df_ks = df.sort_values(y_pre, ascending=False).reset_index(drop=True)
    df_ks['rank'] = np.floor((df_ks.index / len(df_ks) * num))
    df_ks['set_1'] = 1
    # 2.统计结果
    result_ks = pd.DataFrame()
    result_ks['group_sum'] = df_ks.groupby('rank')['set_1'].sum()
    result_ks['group_min'] = df_ks.groupby('rank')[y_pre].min()
    result_ks['group_max'] = df_ks.groupby('rank')[y_pre].max()
    result_ks['group_mean'] = df_ks.groupby('rank')[y_pre].mean()
    # 3.最后一行添加total汇总数据
    result_ks.loc['total', 'group_sum'] = df_ks['set_1'].sum()
    result_ks.loc['total', 'group_min'] = df_ks[y_pre].min()
    result_ks.loc['total', 'group_max'] = df_ks[y_pre].max()
    result_ks.loc['total', 'group_mean'] = df_ks[y_pre].mean()
    # 4.好用户统计
    result_ks['good_sum'] = df_ks[df_ks[y_true] == good].groupby('rank')['set_1'].sum()
    result_ks.good_sum.replace(np.nan, 0, inplace=True)
    result_ks.loc['total', 'good_sum'] = result_ks['good_sum'].sum()
    result_ks['good_percent'] = result_ks['good_sum'] / result_ks.loc['total', 'good_sum']
    result_ks['good_percent_cum'] = result_ks['good_sum'].cumsum() / result_ks.loc['total', 'good_sum']
    # 5.坏用户统计
    result_ks['bad_sum'] = df_ks[df_ks[y_true] == bad].groupby('rank')['set_1'].sum()
    result_ks.bad_sum.replace(np.nan, 0, inplace=True)
    result_ks.loc['total', 'bad_sum'] = result_ks['bad_sum'].sum()
    result_ks['bad_percent'] = result_ks['bad_sum'] / result_ks.loc['total', 'bad_sum']
    result_ks['bad_percent_cum'] = result_ks['bad_sum'].cumsum() / result_ks.loc['total', 'bad_sum']
    # 6.计算ks值
    result_ks['diff'] = result_ks['bad_percent_cum'] - result_ks['good_percent_cum']
    # 7.更新最后一行total的数据
    result_ks.loc['total', 'bad_percent_cum'] = np.nan
    result_ks.loc['total', 'good_percent_cum'] = np.nan
    result_ks.loc['total', 'diff'] = result_ks['diff'].max()
    result_ks['good_cum'] = result_ks['good_sum'].cumsum()
    result_ks['bad_cum'] = result_ks['bad_sum'].cumsum()
    result_ks['rate'] = result_ks['bad_cum'] / (result_ks['bad_cum'] + result_ks['good_cum']) * 100
    result_ks = result_ks.reset_index()
    return result_ks


def ksCurve(df, num=10, model='train'):
    # 防止中文乱码
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.clf()
    ks_value = df['diff'].max()
    # 获取绘制曲线所需要的数据
    x_curve = range(num + 1)
    y_curve1 = [0] + list(df['bad_percent_cum'].values[:-1])
    y_curve2 = [0] + list(df['good_percent_cum'].values[:-1])
    y_curve3 = [0] + list(df['diff'].values[:-1])
    # 获取绘制ks点所需要的数据
    df_ks_max = df[df['diff'] == ks_value]
    x_point = [df_ks_max['rank'].values[0] + 1, df_ks_max['rank'].values[0] + 1]
    y_point = [df_ks_max['bad_percent_cum'].values[0], df_ks_max['good_percent_cum'].values[0]]
    # 绘制曲线
    plt.plot(x_curve, y_curve1, label='bad', linewidth=2)
    plt.plot(x_curve, y_curve2, label='good', linewidth=2)
    plt.plot(x_curve, y_curve3, label='diff', linewidth=2)
    # 标记ks
    plt.plot(x_point, y_point, label='ks - {:.2f}'.format(ks_value), color='r', marker='o', markerfacecolor='r',
             markersize=5)
    plt.scatter(x_point, y_point, color='r')
    plt.legend()
    plt.savefig(model + "_ks.jpg")
    return ks_value


def plotKS(score, label, num=10, good=0, bad=1, model='train'):
    df = ksDf(score=score, label=label, num=num, good=good, bad=bad)
    ks = ksCurve(df, num=num, model=model)
    print("ks = ", ks)