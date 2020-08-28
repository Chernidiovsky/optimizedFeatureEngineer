# -*- coding: utf-8 -*-
from scipy.stats import pearsonr
from collections import OrderedDict
import random
import time
import pandas as pd
import numpy as np
import sys


def _reachLimit(func):
    def wrapper(c):
        temp = func(c)
        if len(c._tempUsedFeatures) >= c._featuresQuanLimitation or (time.time() - c._startTime) >= c._timeLimitation:
            if len(c._tempUsedFeatures) >= c._featuresQuanLimitation:
                print("Maximum features limit reach!")
            if (time.time() - c._startTime) >= c._timeLimitation:
                print("Time's up!")
            print('*-*' * 50)
            print("best score:{}".format(c._score))
            print('best {} features combination: {}'.format(c._featuresQuanLimitation, c._TemplUsedFeatures))
            sys.exit()
        return temp

    return wrapper


class LrsSaRgssCombination:
    def __init__(self, clf, df, recordFolder, columnName, start, label, process, direction, lossFunction,
                 featuresQuanLimitation, featureEachRound, timeLimitation, featureEachRoundRandom, sampleRatio,
                 sampleMode, sampleState, fitParams, validateFunction, potentialAdd, crossMethod, coherenceThreshold):
        self._clf = clf
        self._df = df
        self._recordFolder = recordFolder
        self._columnName = columnName
        self._tempUsedFeatures = start
        self._label = label
        self.process = process
        self._direction = direction
        self._lossFunction = lossFunction
        self._featuresQuanLimitation = featuresQuanLimitation
        self._featureEachRound = featureEachRound
        self._timeLimitation = timeLimitation * 60
        self._featureEachRoundRandom = featureEachRoundRandom
        if sampleRatio > 1:
            self._sampleRatio = 1
        elif sampleRatio <= 0:
            print("sample ratio should be positive, the set up sample ratio is wrong")
            sys.exit()
        else:
            self._sampleRatio = sampleRatio
        self._sampleMode = sampleMode
        self._sampleState = sampleState
        self._fitParams = fitParams
        self._validateFunction = validateFunction
        self._potentialAdd = potentialAdd
        self._crossMethod = crossMethod
        self._coherenceThreshold = coherenceThreshold

        self._startCol = []
        self._bestFeature = self._tempUsedFeatures[:]
        self._startTime = time.time()
        self._score = 0.
        self._greedyScore = 0.
        self.bestScore = 0.
        self.remain = ''
        self._first = False
        self.delete = ''

    def _evaluate(self, a, b):
        return a > b if self._direction == 'ascend' else a < b

    def select(self):
        self._startTime = time.time()
        if self._direction == 'ascend':
            self._score, self._greedyScore = -np.inf, np.inf
        else:
            self._score, self._greedyScore = np.inf, -np.inf
        self.remain = ''
        self._first = True

        while self._evaluate(self._score, self._greedyScore) or self._first:
            print('test performance of initial features combination')
            self.bestScore, self._bestFeature = self._score, self._tempUsedFeatures[:]

            # if self._tempUsedFeatures[:] != [] and self._first:
            #     self._validation(self._tempUsedFeatures[:], str(0), 'baseline')

            if self.process[0]:
                self._greedy()
            self._scoreUpdate()
            self._greedyScore = self.bestScore
            print('random select starts with:\n {0}\n score: {1}'.format(self._bestFeature, self._greedyScore))
            if self.process[1] is True:
                self._myRandom()
            if self.process[2] and self._first:
                print('small cycle cross')
                n = True
                while self._scoreUpdate() or n:
                    self._crossTermSearch(self._bestFeature, self._bestFeature)
                    n = False
                if self._greedyScore == self._score:
                    print('medium cycle cross')
                    n = True
                    while self._scoreUpdate() or n:
                        self._crossTermSearch(self._columnName, self._bestFeature)
                        n = False
                if self._greedyScore == self._score:
                    print('large cycle cross')
                    n = True
                    while self._scoreUpdate() or n:
                        self._crossTermSearch(self._columnName, self._columnName)
                        n = False
            self._first = False
            self._scoreUpdate()

        print('{0}\nbest score:{1}\nbest features combination: {2}'.format('*-*' * 50, self.bestScore, self._bestFeature))
        with open(self._recordFolder, 'a') as f:
            f.write('{0}\nbest score:{1}\nbest features combination: {2}'.format('*-*' * 50, self.bestScore, self._bestFeature))
        return self._bestFeature

    def _validation(self, selectCol, num, addFeature):
        self.checkLimit()
        selectCol = list(OrderedDict.fromkeys(selectCol))
        self._sampleState += self._sampleMode
        if self._sampleRatio < 1:
            tempDf = self._df.sample(frac=self._sampleRatio, random_state=self._sampleState).reset_index(drop=True)
        else:
            tempDf = self._df
        X, y = tempDf, tempDf[self._label]
        totalTest = self._validateFunction(X, y, selectCol, self._clf, self._lossFunction)
        print('Mean loss: {}'.format(totalTest))
        if self._evaluate(np.mean(totalTest), self._score):
            print("_evaluate")
            cc = [0]
            if self._coherenceThreshold != 1:
                tmpCol = selectCol[:]
                tmpCol.remove(addFeature)
                cc = [pearsonr(self._df[addFeature], self._df[ct])[0] for ct in tmpCol]
            if np.abs(np.max(cc)) < self._coherenceThreshold:
                with open(self._recordFolder, 'a') as f:
                    f.write('{0}  {1}  {2}:\n{3}\t{4}\n'.format(num, addFeature, np.abs(np.max(cc)), np.round(np.mean(totalTest), 6), selectCol[:]))
                    f.write('*{}\n'.format(np.round(np.mean(totalTest), 6)))
                    for s in selectCol[:]:
                        f.write('{} '.format(s))
                    f.write('\n')

                self._tempUsedFeatures, self._score = selectCol[:], np.mean(totalTest)
                if num == 'reverse':
                    self.delete = addFeature
                    print("%s is picked in backward." % addFeature)
                else:
                    self.remain = addFeature
                    print("%s is picked in forward." % addFeature)

    def _greedy(self):
        col = self._columnName[:]
        print('{0}{1}{2}'.format('-' * 20, 'start greedy', '-' * 20))
        for i in self._tempUsedFeatures:
            print(i)
            try:
                col.remove(i)
            except:
                pass
        self.delete = ''
        self.bestScore, self._bestFeature = self._score, self._tempUsedFeatures[:]
        _featureEachRoundLoop = 0
        _featureEachRoundMaxLoop = 10
        _loopCount = 0

        # stop when no improve for the last round and no potential add feature
        while self._startCol != self._tempUsedFeatures or self._potentialAdd != [] or _featureEachRoundLoop < _featureEachRoundMaxLoop:
            _loopCount += 1
            if self._startCol == self._tempUsedFeatures[:]:  # no improve
                if _featureEachRoundLoop < _featureEachRoundMaxLoop:
                    _featureEachRoundLoop += 1  # set maximum loop number for random select self._featureEachRound of features each round
                else:
                    self._scoreUpdate()
                    if self._direction == 'ascend':
                        self._score *= 0.95  # Simulate Anneal Arithmetic, step back a bit, the value need to be change
                    else:
                        self._score /= 0.95
                    self._tempUsedFeatures.append(self._potentialAdd[0])
            else:
                _featureEachRoundLoop = 0
            print('{0} {1} round {2}'.format('*' * 20, len(self._tempUsedFeatures) + 1, '*' * 20))
            if self.remain in col:
                col.remove(self.remain)
            if self.delete != '':
                col.append(self.delete)
            self._startCol = self._tempUsedFeatures[:]
            if len(col) > self._featureEachRound > 0:
                if self._featureEachRoundRandom:  # random select _featureEachRound features from all
                    useCol = np.random.choice(col, self._featureEachRound, replace=False).tolist()
                else:  # select _featureEachRound features from all chunk by chunk
                    useCol = col[_featureEachRoundLoop * self._featureEachRound: (_featureEachRoundLoop + 1) * self._featureEachRound]
                _featureEachRoundMaxLoop = len(col) // self._featureEachRound
            else:
                useCol = col
                _featureEachRoundMaxLoop = 0

            # forward sequence selection add one each round
            for sub, i in enumerate(useCol):
                print(i)
                print('{}/{}'.format(sub, len(useCol)))
                selectCol = self._startCol[:]
                selectCol.append(i)
                self._validation(selectCol, str(1 + sub), i)

            # backward sequence selection, -1 because the last 1 is just selected
            for sr, i in enumerate(self._tempUsedFeatures[:-1]):
                deleteCol = self._tempUsedFeatures[:]
                # if i in deleteCol:
                #     deleteCol.remove(i)
                print(i)
                print('reverse {}/{}'.format(sr, len(self._tempUsedFeatures[:-1])))
                self._validation(deleteCol, 'reverse', i)

            for i in self._tempUsedFeatures:
                if i in self._potentialAdd:
                    self._potentialAdd.remove(i)
        print('{0}{1}{2}'.format('-' * 20, 'complete greedy', '-' * 20))

    def _myRandom(self):
        self._scoreUpdate()
        col = self._columnName[:]
        print('{0}{1}{2}'.format('-' * 20, 'start random', '-' * 20))
        for i in self._bestFeature:
            if i in col:
                col.remove(i)
        random.seed(a=self._sampleState)
        for t in range(3, 9):
            if t < len(col):
                print('add {} features'.format(t))
                for i in range(50):
                    selectCol = random.sample(col, t)
                    recordAdd = selectCol[:]
                    for add in self._bestFeature:
                        selectCol.append(add)
                    self._validation(selectCol, str(i), str(recordAdd))
        print('{0}{1}{2}'.format('-' * 20, 'complete random', '-' * 20))

    @_reachLimit
    def checkLimit(self):
        return True

    @_reachLimit
    def _scoreUpdate(self):
        if self._direction == 'ascend':
            start = -np.inf
        else:
            start = np.inf
        if self._score == start:
            return True
        elif self._evaluate(self._score, self.bestScore):
            self.bestScore, self._bestFeature = self._score, self._tempUsedFeatures[:]
            return True
        return False

    def _crossTermSearch(self, col1, col2):
        self._scoreUpdate()
        Effective = []
        crossCount = 0
        for c1 in col1:
            for c2 in col2[::-1]:
                for op in self._crossMethod.keys():
                    print('{}/{}'.format(crossCount, len(self._crossMethod.keys()) * len(col1) * len(col2)))
                    crossCount += 1
                    newColName = "({}{}{})".format(c1, op, c2)
                    self._df[newColName] = self._crossMethod[op](self._df[c1], self._df[c2])
                    selectCol = self._bestFeature[:]
                    selectCol.append(newColName)
                    try:
                        self._validation(selectCol, 'cross term', newColName)
                    except:
                        pass
                    if self._scoreUpdate():
                        Effective.append(newColName)
                    else:
                        self._df.drop(newColName, axis=1, inplace=True)
        if self.remain in Effective:
            Effective.remove(self.remain)
        self._columnName.append(self.remain)


class Select(object):
    """This is a class for sequence/random/crossterm features selection

    The functions needed to be called before running include:

        importDF(pd.dataframe, str) - import you complete dataset including the label column

        importlossFunction(func, str) - import your self define loss function,
                                        eq. logloss, accuracy, etc

        initialFeatures(list) - Initial your starting features combination,
                                if the initial features combination include
                                all features, the backward sequence searching
                                will run automatically

        initialNonTrainableFeatures(list) - Initial the non-trainable features

        importcrossMethod(dict) - Import your cross method, eq. +, -, *, /,
                                  can be as complicate as you want, this requires
                                  setup if Cross = True

        addPotentialFeatures(list) - give some strong features you think might
                                     be useful. It will force to add into the
                                     features combination once the sequence
                                     searching doesn't improve

        setCCThreshold(float) - Set the maximum correlation coefficient between each
                                features

        run(func) - start selecting features
    """

    def __init__(self, sequencePro=True, randomPro=True, crossPro=True, logfile='record.log'):
        self.sequence = sequencePro
        self.random = randomPro
        self.cross = crossPro

        self._logfile = logfile
        self._df = pd.DataFrame()
        self._label = ""
        self._crossMethod = {}
        self._modelScore = 0.
        self._direction = None
        self._start = []
        self._nonTrainableFeatures = []
        self.columnName = []
        self._featureEachRound = 100000000
        self._featureEachRoundRandom = False
        self._potentialAdd = []
        self._coherenceThreshold = 1
        self._featuresLimit = np.inf
        self._timeLimit = np.inf
        self._sampleRatio = 1
        self._sampleState = 0
        self._sampleMode = 1
        self.clf = None
        self.fitParams = {}

    def importDf(self, df, label):
        """Import pandas dataframe to the class

        Args:
            df: pandas dataframe include all features and label.
            label: str, label name
        """
        self._df = df
        self._label = label

    def importCrossMethod(self, crossMethod):
        """Import a dictionary with different cross function

        Args:
            crossMethod: dict, dictionary with different cross function
        """
        self._crossMethod = crossMethod

    def importLossFunction(self, modelScore, direction):
        """Import the loss function

        Args:
            modelScore: the function to calculate the loss result
                        with two input series
            direction: str, ‘ascent’ or descent, the way you want
                       the score to go
        """
        self._modelScore = modelScore
        self._direction = direction

    def initialFeatures(self, features):
        """Initial your starting features combination

        Args:
            features: list, the starting features combination
        """
        self._start = features

    def initialNonTrainableFeatures(self, features):
        """Setting the nontrainable features, eq. user_id

        Args:
            features: list, the nontrainable features
        """
        self._nonTrainableFeatures = features

    def generateCol(self, key=None, selectStep=1):
        """ for getting rid of the useless columns in the dataset
            key: None of list of key word
        """
        self.columnName = self._df.columns.tolist()[:]
        for i in self._nonTrainableFeatures:
            if i in self.columnName:
                self.columnName.remove(i)
        if isinstance(key, list):
            self.columnName = []
            for k in key:
                self.columnName += [i for i in self.columnName if k in i]
        elif isinstance(key, str):
            self.columnName = [i for i in self.columnName if key in i]
        self.columnName = self.columnName[::selectStep]  # ???

    def setFeatureEachRound(self, ser, featureEachRoundRandom):
        """for speeding up adding features each round
        Args:
           ser: random select ser features each round
           featureEachRoundRandom: bool, if it is true, ser features will be selected randomly
                                   from features pool, if false, they will be selected chunk
                                   by chunk
        """
        self._featureEachRound = ser
        self._featureEachRoundRandom = featureEachRoundRandom

    def addPotentialFeatures(self, features):
        """give some strong features you think might be useful.

        Args:
            features: list, the strong features that not in InitialFeatures
        """
        self._potentialAdd = features

    def setCCThreshold(self, cc):
        """Set the maximum correlation coefficient between each features

        Args:
            cc: float, the upper bound of correlation coefficient
        """
        self._coherenceThreshold = cc

    def setFeaturesLimit(self, featuresLimit):
        """Set the features quantity limitation, when selected features reach
           the quantity limitation, the algorithm will exit

        Args:
            featuresLimit: int, the features quantity limitation
        """
        self._featuresLimit = featuresLimit

    def setTimeLimit(self, timeLimit):
        """Set the running time limitation, when the running time
           reach the time limit, the algorithm will exit

        Args:
            timeLimit: double, the maximum time in minutes
        """
        self._timeLimit = timeLimit

    def setSample(self, ratio, sampleState=0, sampleMode=1):
        """Set the sample of all data

        Args:
            ratio: double, sample ratio
            sampleState: int, seed
            sampleMode: positive int, if 0, every time they
                        sample the same subset, default = 1
        """
        self._sampleRatio = ratio
        self._sampleState = sampleState
        self._sampleMode = sampleMode

    def setClassifier(self, clf, fitParams=None):
        """Set the classifier and its fitParams

        Args:
            clf: estimator object, defined algorithm to train and evaluate features
            fitParams: dict, optional, parameters to pass to the fit method
        """
        self.clf = clf
        if fitParams is None:
            self.fitParams = fitParams

    def run(self, validate):
        """start running the selecting algorithm

        Args:
            validate: validation method, eq. kfold, last day, etc
        """
        with open(self._logfile, 'a') as f:
            f.write('\n{}\n%{}%\n'.format('Start!', '-' * 60))
        print("Features Quantity Limit: {}".format(self._featuresLimit))
        print("Time Limit: {} min(s)".format(self._timeLimit))
        print(self._featureEachRound)
        a = LrsSaRgssCombination(df=self._df, clf=self.clf, featureEachRound=self._featureEachRound,
                                 featureEachRoundRandom=self._featureEachRoundRandom, recordFolder=self._logfile,
                                 lossFunction=self._modelScore, label=self._label, columnName=self.columnName[:],
                                 start=self._start[:], crossMethod=self._crossMethod, potentialAdd=self._potentialAdd,
                                 process=[self.sequence, self.random, self.cross], direction=self._direction,
                                 validateFunction=validate, coherenceThreshold=self._coherenceThreshold,
                                 featuresQuanLimitation=self._featuresLimit, timeLimitation=self._timeLimit,
                                 sampleRatio=self._sampleRatio, sampleState=self._sampleState,
                                 sampleMode=self._sampleMode, fitParams=self.fitParams)
        best_feature_comb = a.select()
        with open(self._logfile, 'a') as f:
            f.write('\n{}\n{}\n%{}%\n'.format('Done', self._start, '-' * 60))
        return best_feature_comb
