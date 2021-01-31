import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import decomposition, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from utils import *

def bestRandomForest(X, y, split=0.7, ntrials=100):
    means = np.zeros(ntrials, )
    max_accuracy= 0
    for trial in range(ntrials):
        xTr, yTr, xTe, yTe, trIdx, teIdx = trteSplitEven(X, y, split, trial)
        # Train
        scaler = RobustScaler()
        scaler.fit(xTr)
        xTr = scaler.transform(xTr)
        xTe = scaler.transform(xTe)
        forest = RandomForestClassifier(class_weight="balanced")
        n_estimators = [100, 300, 600]
        max_depth = [3, xTr.shape[1] / 2 + 1, 25, 100, 300]
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 10]
        hyperF = dict(n_estimators=n_estimators, max_depth=max_depth,
                      min_samples_split=min_samples_split,
                      min_samples_leaf=min_samples_leaf)

        gridF = GridSearchCV(forest, hyperF, cv=3, verbose=1, n_jobs=-1)
        gridF.fit(xTr, yTr)
        best_params = gridF.best_params_
        print(best_params)
        # Predict
        yPr = gridF.predict(xTe)
        # Compute classification error
        print("Trial:", trial, "Accuracy", "%.3g" % (100 * np.mean((yPr == yTe).astype(float))))
        means[trial] = 100 * np.mean((yPr == yTe).astype(float))
        if means[trial]>max_accuracy:
            max_accuracy = means[trial]
            best_classifier = gridF

    print("best accuracy is ", max_accuracy)
    print("best parameters are ", best_classifier.get_params())
    return best_classifier



def testClassifier(classifier, X, y, split=0.7, ntrials=100):

    means = np.zeros(ntrials, )
    for trial in range(ntrials):
        xTr, yTr, xTe, yTe, trIdx, teIdx = trteSplitEven(X, y, split, trial)
        # Train
        scaler = RobustScaler()
        scaler.fit(xTr)
        xTr = scaler.transform(xTr)
        xTe = scaler.transform(xTe)
        trained_classifier = classifier.trainClassifier(xTr, yTr)
        # Predict
        yPr = trained_classifier.classify(xTe)
        # Compute classification error
        #if trial % 10 == 0:
        print("Trial:", trial, "Accuracy", "%.3g" % (100 * np.mean((yPr == yTe).astype(float))))

        means[trial] = 100 * np.mean((yPr == yTe).astype(float))

    print("Final mean classification accuracy ", "%.3g" % (np.mean(means)), "with standard deviation",
          "%.3g" % (np.std(means)))


def trteSplitEven(X, y, pcSplit, seed=None):
    labels = np.unique(y)
    xTr = np.zeros((0, X.shape[1]))
    xTe = np.zeros((0, X.shape[1]))
    yTe = np.zeros((0,), dtype=int)
    yTr = np.zeros((0,), dtype=int)
    trIdx = np.zeros((0,), dtype=int)
    teIdx = np.zeros((0,), dtype=int)
    np.random.seed(seed)
    for label in labels:
        classIdx = np.where(y == label)[0]
        NPerClass = len(classIdx)
        Ntr = int(np.rint(NPerClass * pcSplit))
        idx = np.random.permutation(NPerClass)
        trClIdx = classIdx[idx[:Ntr]]
        teClIdx = classIdx[idx[Ntr:]]
        trIdx = np.hstack((trIdx, trClIdx))
        teIdx = np.hstack((teIdx, teClIdx))
        # Split data
        xTr = np.vstack((xTr, X[trClIdx, :]))
        yTr = np.hstack((yTr, y[trClIdx]))
        xTe = np.vstack((xTe, X[teClIdx, :]))
        yTe = np.hstack((yTe, y[teClIdx]))

    return xTr, yTr, xTe, yTe, trIdx, teIdx


df = pd.read_csv("TrainOnMe.csv", na_values=["?"])
df = df.drop(df.index[606])
df['y'] = pd.factorize(df['y'])[0].astype(np.uint16)
df = pd.get_dummies(df, columns=['x5', 'x6'])
features = df[['x1', 'x2', 'x3', 'x4', 'x5_False', 'x5_True', 'x6_A', 'x6_B', 'x6_C', 'x6_D', 'x6_E', 'x6_F', 'x6_Fx', 'x7', 'x8', 'x9', 'x10']].to_numpy(dtype='float', na_value=np.nan)
col_mean = np.nanmean(features, axis=0)
inds = np.where(np.isnan(features))
features[inds] = np.take(col_mean, inds[1])
label = df[['y']].to_numpy(dtype='int')
label = np.reshape(label, (1000,))


#   testing a bunch of classifiers
testClassifier(RandForestClassifier(), features, label, split=0.7)
#testClassifier(DecisionTreeClassifier(), features, label, split=0.7)
#testClassifier(BoostClassifier(RandForestClassifier(), T=10), features, label, split=0.7)
#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), features, label, split=0.7)
#testClassifier(BoostClassifier(BayesClassifier(), T=10), features, label, split=0.7)
#testClassifier(SVMClassifier(), features, label, split=0.7)
#testClassifier(NNClassifier(), features, label, split=0.7)

#   the best classifier found using accuracy as a measurement is the random forest one

# tuning the hyperparameters through a CV grid search and getting back the classifier with the highest accuracy
classifier = bestRandomForest(features, label, split=0.7)


df = pd.read_csv("EvaluateOnMe.csv", na_values=["?"])
df = pd.get_dummies(df, columns=['x5', 'x6'])
features = df[['x1', 'x2', 'x3', 'x4', 'x5_False', 'x5_True', 'x6_A', 'x6_B', 'x6_C', 'x6_D', 'x6_E', 'x6_F', 'x6_Fx', 'x7', 'x8', 'x9', 'x10']].to_numpy(dtype='float', na_value=np.nan)
yPred= classifier.predict(features)
trans = {0: 'Bob', 1: 'Atsuto', 2: 'Jörg'}
yPred = list(yPred)
for i in range(len(yPred)):
    for key in trans.keys():
        if key==yPred[i]:
            yPred[i]= trans[key]
df = pd.DataFrame(yPred)
df.to_csv('list.csv', index=False, header=False)

