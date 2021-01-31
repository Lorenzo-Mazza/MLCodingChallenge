import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import decomposition, tree, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)


# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    Nclasses - the number of different classes
# out:  yPred - N vector of class predictions for test points

def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts, Nclasses))

        # TODO: implement classification when we have trained several classifiers!
        # here we can do it by filling in the votes vector with weighted votes
        # ==========================
        for i_iter in range(Ncomps):
            labels = classifiers[i_iter].classify(X)
            # labels = np.reshape(labels,(-1,1))
            for idx in range(Npts):
                votes[idx, labels[idx]] += alphas[i_iter]
        # ==========================
        # one way to compute yPred after accumulating the votes
        return np.argmax(votes, axis=1)


# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#                   T - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
#              alphas - (maximum) length T Python list of vote weights
def trainBoost(base_classifier, X, labels, T=10):
    # these will come in handy later on
    Npts, Ndims = np.shape(X)

    classifiers = []  # append new classifiers to this list
    alphas = []  # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts, 1)) / float(Npts)

    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))

        # do classification for each point
        vote = classifiers[-1].classify(X)
        # ==========================
        delta = np.where(vote == labels, 0, 1)
        error = np.dot(delta, wCur)
        alpha = 0.5 * ((np.log(1 - error)) - np.log(error))
        alphas.append(alpha)  # you will need to append the new alpha
        pos_index = np.where(vote == labels)[0]
        pos = np.sum(wCur[pos_index])
        neg_index = np.where(vote != labels)[0]
        neg = np.sum(wCur[neg_index])
        Z = pos * np.exp(-alpha) + neg * np.exp(alpha)
        wCur[pos_index] = wCur[pos_index] * np.exp(-alpha) / Z  # update weights
        wCur[neg_index] = wCur[neg_index] * np.exp(alpha) / Z  # update weights
        # ==========================

    return classifiers, alphas


class DecisionTreeClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, Xtr, yTr, W=None):
        rtn = DecisionTreeClassifier()
        rtn.classifier = tree.DecisionTreeClassifier(max_depth=Xtr.shape[1] / 2 + 1)
        if W is None:
            rtn.classifier.fit(Xtr, yTr)
        else:
            rtn.classifier.fit(Xtr, yTr, sample_weight=W.flatten())
        rtn.trained = True
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)


class RandForestClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, Xtr, yTr, W=None):
        rtn = RandForestClassifier()
        rforest = RandomForestClassifier(max_depth=1000)
        rtn.classifier= rforest
        rtn.classifier.fit(Xtr, yTr)
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)


class NNClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, Xtr, yTr, W=None):
        rtn = NNClassifier()
        rtn.classifier = MLPClassifier(solver='adam',max_iter=10000, warm_start=True, alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

        if W is None:
            rtn.classifier.fit(Xtr, yTr)
        else:
            rtn.classifier.coefs_= W
            rtn.classifier.fit(Xtr, yTr)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)


class SVMClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, Xtr, yTr, W=None):
        rtn = SVMClassifier()
        rtn.classifier = svm.SVC(decision_function_shape='ovo')
        if W is None:
            rtn.classifier.fit(Xtr, yTr)
        else:
            rtn.classifier.fit(Xtr, yTr, sample_weight=W.flatten())
        rtn.trained = True
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)


class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)


# in:      X - N x d matrix of M data points
#      prior - C x 1 matrix of class priors
#         mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
# out:     h - N vector of class predictions for test points

def classifyBayes(X, prior, mu, sigma):
    Npts = X.shape[0]
    Nclasses, Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))
    determinant = np.zeros(Nclasses)
    inverseSigma = np.zeros((Nclasses, Ndims, Ndims))
    # compute determinant and inverse matrix of sigma for each class jdx
    for jdx in range(Nclasses):
        determinant[jdx] = np.prod(np.diag(sigma[jdx]))
        for idx in range(Ndims):
            inverseSigma[jdx][idx][idx] = 1 / sigma[jdx][idx][idx]
    # compute log posterior for each datapoint X[idx] and each class jdx
    for idx in range(Npts):
        for jdx in range(Nclasses):
            difference = X[idx] - mu[jdx]
            ris = 0.5 * (np.dot(np.dot(difference, inverseSigma[jdx]), np.matrix.transpose(difference)))
            logProb[jdx][idx] = -0.5 * np.log(determinant[jdx]) - ris + prior[jdx]
    # ==========================
    # one possible way of finding max a-posteriori once
    # you have computed the log posterior
    h = np.argmax(logProb, axis=0)
    return h


# NOTE: you do not need to handle the W argument for this part!
# in: labels - N vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(labels, W):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts, 1)) / Npts
    else:
        assert (W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)
    prior = np.zeros((Nclasses, 1))

    for jdx, class_match in enumerate(classes):
        idx = np.where(labels == class_match)[0]
        prior[jdx] = np.sum(W[idx]) / np.sum(W)

    return prior


# in:      X - N x d matrix of N data points
#     labels - N vector of class labels
#          W - N x 1 matrix of weights
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
def mlParams(X, labels, W):
    assert (X.shape[0] == labels.shape[0])

    Npts, Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)
    if W is None:
        W = np.ones((Npts, 1)) / Npts

    mu = np.zeros((Nclasses, Ndims))
    sigma = np.zeros((Nclasses, Ndims, Ndims))

    for jdx, class_match in enumerate(classes):
        idx = np.where(labels == class_match)[0]  # idx= array of indexes of the elements classified in class jdx
        Nk = len(idx)
        xlc = X[idx, :]  # xlc = array [K x D] of all the elements in class jdx
        wlc = W[idx]  # wlc = array [K x 1] of all the weights for class jdx
        for feature in range(Ndims):
            mu[jdx, feature] = np.nansum(np.reshape(wlc, (1, -1)) * xlc[:, feature]) / np.sum(
                wlc)  # compute mu of each feature for class jdx
            sq = np.square(xlc[:, feature] - mu[jdx, feature])  # compute sigma-k [m,m]
            rs = np.reshape(wlc, (1, -1))
            r = np.nansum(rs * sq)
            # r = np.dot(rs, sq)
            sigma[jdx][feature][feature] = r / np.sum(wlc)
    #    d = xlc - mu[jdx,:]
    #    prod = np.dot(np.matrix.transpose(d),d)
    #    prod = np.diag(np.diag(prod))
    #    sigma[jdx] = prod/Nk

    return mu, sigma
