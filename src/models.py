import warnings

warnings.filterwarnings("ignore")

# tree based classifier
import xgboost as xgb
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.gaussian_process.gpc import GaussianProcessClassifier

# ensemble classifiers
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.forest import RandomForestClassifier

# linear classifiers
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron

from sklearn import metrics

import pandas as pd
import numpy as np


class ClassificationModels:
    def __init__(self):
        self.models = {
            "logistic_regression": self._logistic_regression,
            "random_forest": self._random_forest,
            "svc": self._svc,
            "knn": self._knn,
        }

    def __call__(self, model, X_train, y_train):
        if model not in self.models:
            raise Exception("Model not implemented")
        else:
            return self.models[model](X_train, y_train)

    @staticmethod
    def _logistic_regression(X_train, y_train):
        return LogisticRegression(class_weight="balanced").fit(X_train, y_train)

    @staticmethod
    def _random_forest(X_train, y_train):
        return RandomForestClassifier().fit(X_train, y_train)

    @staticmethod
    def _svc(X_train, y_train):
        return SVC().fit(X_train, y_train)

    @staticmethod
    def _knn(X_train, y_train):
        return KNeighborsClassifier().fit(X_train, y_train)
