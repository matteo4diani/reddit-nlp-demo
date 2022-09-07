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

##################
# POS Vectorizer #
##################

def get_pos_tag_count(text : str) -> dict[str, float]:
    tags = nltk.pos_tag(nltk.word_tokenize(text.lower()))
    counts = Counter(tag for word, tag in tags)
    return counts

class PosTagVectorizer(BaseEstimator, TransformerMixin):
    """
    Custom vectorizer.
    Transforms a list of strings in a matrix of Part-Of-Speech counts.
    Uses DictVectorizer by default.
    """
    def __init__(self, vectorizer=None):
        super().__init__()
        if vectorizer:
            self.vectorizer = vectorizer
        else:
            self.vectorizer = DictVectorizer()
    
    def fit(self, X, y=None):
        self.vectorizer.fit(map(get_pos_tag_count, X.body))
        return self
    
    def transform(self, X, y=None):
        return self.vectorizer.transform(map(get_pos_tag_count, X.body))
    
    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()
    
    
########################
# Subreddit Vectorizer #
########################

def get_subreddit_count(subreddits : str) -> dict[str, int]:
    subs = subreddits.split()
    counts = Counter(subreddit for subreddit in subs)
    return counts


class SubredditVectorizer(BaseEstimator, TransformerMixin):
    """
    Custom vectorizer.
    Transforms a list of strings of subreddits (separated by whitespace) in a matrix of comments-in-subreddit counts.
    Uses DictVectorizer by default.
    """
    def __init__(self, vectorizer=None):
        super().__init__()
        if vectorizer:
            self.vectorizer = vectorizer
        else:
            self.vectorizer = DictVectorizer()
    
    def fit(self, X, y=None):
        self.vectorizer.fit(map(get_subreddit_count, X.subreddit))
        return self
    
    def transform(self, X, y=None):
        return self.vectorizer.transform(map(get_subreddit_count, X.subreddit))
    
    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()
    
####################
# Epoch Vectorizer #
####################

def to_date(epoch):
    return datetime.utcfromtimestamp(epoch)

def to_slug(epoch):
    date = to_date(float(epoch))
    ordinal = date.weekday()*24 + date.hour
    weekday = calendar.day_name[date.weekday()]
    hour = date.hour
    return f'{ordinal:03}_{weekday.upper()}_{hour:02}'

def get_comment_time_count(comment_times : str) -> dict[str, int]:
    times = comment_times.split(',')
    slugged_times = map(to_slug, times)
    counts = Counter(time for time in slugged_times)
    return counts

class EpochVectorizer(BaseEstimator, TransformerMixin):
    """
    Custom vectorizer.
    Transforms a list of strings of epoch times (assuming UTC, separated by commas) in a matrix of counts binned by weekday+hour (bins are sorted chronologically).
    Uses DictVectorizer by default.
    """
    def __init__(self, vectorizer=None):
        super().__init__()
        if vectorizer:
            self.vectorizer = vectorizer
        else:
            self.vectorizer = DictVectorizer(sort=True)
    
    def fit(self, X, y=None):
        self.vectorizer.fit(map(get_comment_time_count, X.created_utc))
        return self
    
    def transform(self, X, y=None):
        return self.vectorizer.transform(map(get_comment_time_count, X.created_utc))
    
    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()
    
#######################
# Username Vectorizer #
#######################

class UsernameVectorizer(BaseEstimator, TransformerMixin):
    """
    Custom vectorizer.
    Transforms a list of usernames in a matrix of character counts (case sensitive), adding the total length of the username as last column.
    Uses DictVectorizer by default.
    """
    def __init__(self, vectorizer=None, ngram_range=(1,1), lowercase=True):
        super().__init__()
        if vectorizer:
            self.vectorizer = vectorizer
        else:
            self.vectorizer = CountVectorizer(analyzer='char',
                                              ngram_range=ngram_range,
                                              lowercase=lowercase)
            self.ngram_range = ngram_range
            self.lowercase = lowercase
    
    def fit(self, X, y=None):
        self.vectorizer.fit(X.author)
        return self
    
    def transform(self, X, y=None):
        transformed = self.vectorizer.transform(X.author)
        return sparse.hstack([transformed,transformed.sum(axis = 1)])

    def get_feature_names_out(self):
        return [*self.vectorizer.get_feature_names_out(), 'length']
    
    
########################
# Text Body Vectorizer #
########################

class BodyVectorizer(BaseEstimator, TransformerMixin):
    """
    Custom vectorizer.
    Wraps a TfidfVectorizer that extracts the 'body' column from a DataFrame
    """
    def __init__(self, vectorizer=None, ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None):
        super().__init__()
        if vectorizer:
            self.vectorizer = vectorizer
            self.max_df = max_df
            self.min_df = min_df
            self.max_features = max_features
        else:
            self.vectorizer = TfidfVectorizer(max_df=max_df, 
                                              min_df=min_df, 
                                              max_features=max_features,
                                              ngram_range=ngram_range)
            self.max_df = max_df
            self.min_df = min_df
            self.max_features = max_features
    
    def fit(self, X, y=None):
        self.vectorizer.fit(X.body)
        return self
    
    def transform(self, X, y=None):
        return self.vectorizer.transform(X.body)

    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()