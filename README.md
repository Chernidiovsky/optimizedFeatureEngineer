# optimizedFeatureEngineering


实现单调最优分箱、前后向差分特征选择、逻辑回归


utilsMonotonicWoeBinning.py
单调最优分箱代码参考自：
https://github.com/jstephenj14/Monotonic-WOE-Binning-Algorithm
这里支持传入qcutNum来设置预分组数。把数据集按大小归纳成qcutNum份，每份视为同一值，以减少自动分箱计算的耗时。

utilsSequenceSelection.py
前后向差分特征选择代码参考自：
https://github.com/duxuhao/Feature-Selection/tree/master/MLFeatureSelection
损失函数参照auc，并对所选特征的数量增加惩罚因子，避免特征过多造成过拟合。
 