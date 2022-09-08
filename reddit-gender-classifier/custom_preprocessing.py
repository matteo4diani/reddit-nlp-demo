from collections.abc import Iterable, Sequence
from collections import Counter
from collections import defaultdict
import copy
import enum
import time
from datetime import datetime
import calendar
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
import re
import nltk

import custom_transformers

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, IsolationForest
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import RocCurveDisplay
from sklearn.pipeline import FeatureUnion

from imblearn import FunctionSampler
from imblearn.pipeline import Pipeline, make_pipeline

import seaborn

#####################
# Utility Functions #
#####################
def outlier_rejection(X, y):
    """Uses an isolation forest model to exclude outliers. Works on a feature matrix and a target array."""
    model = IsolationForest(max_samples='auto', contamination='auto', random_state=0)
    model.fit(X)
    y_pred = model.predict(X)
    return X[y_pred == 1], y[y_pred == 1]

def activity_based_outlier_rejection(X, y):
    """Uses an isolation forest model to exclude outliers. Works on the whole dataframe and filters only on created_utc."""
    epoch_vectorizer = custom_transformers.EpochVectorizer()
    X_epochs = epoch_vectorizer.fit_transform(X)
    model = IsolationForest(max_samples='auto', contamination='auto', random_state=0)
    model.fit(X_epochs)
    y_pred = model.predict(X_epochs)
    return X[y_pred == 1], np.array(y)[y_pred == 1]