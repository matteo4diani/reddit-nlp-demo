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
from datetime import datetime
import csv
import custom_preprocessing

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

def get_dataframes_from_csv(path_to_features, path_to_targets=None):
    """
    Get DataFrames for features and targets
    """
    features_dataframe = pd.read_csv(path_to_features, encoding="utf8")
    
    if not path_to_targets:
        return features_dataframe
    
    targets_dataframe = pd.read_csv(path_to_targets)
    return features_dataframe, targets_dataframe

def get_features_from_dataframe(features_dataframe) -> tuple:
    """
    Get all feature columns as a tuple of lists from DataFrame object
    """
    usernames = list(features_dataframe.author)
    comments = list(features_dataframe.body)
    subreddits = list(features_dataframe.subreddit)
    created_utc = list(features_dataframe.created_utc)
    return usernames, comments, subreddits, created_utc

def get_targets_from_dataframe(features_dataframe, targets_dataframe) -> list:
    """
    Get targets as a list from DataFrame object
    """
    targets_dictionary = dict(zip(targets_dataframe.author, targets_dataframe.gender))
    targets = list(map(lambda a: targets_dictionary[a], features_dataframe.author))
    return targets

def group_dataframe_by_author(features_dataframe):
    """
    Group all features in the dataframe by author.
    """
    return features_dataframe.groupby('author', as_index=False).agg({
                         'subreddit':join_strings, 
                         'body':join_strings, 
                         'created_utc': join_ints})

def join_strings(x : Iterable):
    """
    Join all elements of a list/iterable of strings with a white-space in-between.
    """
    return ' '.join(map(lambda i: str(i), x))

def join_ints(x : Iterable):
    """
    Join all elements of a list/iterable of ints with a comma in-between.
    """
    return ','.join(map(lambda i: str(i), x))

#######################
# Load and clean data #
#######################
def get_training_data():
    training_features_dataframe, training_targets_dataframe = get_dataframes_from_csv("data/train_data.csv", 
                                                                                      "data/train_target.csv")

    training_features_dataframe_groupby_author = group_dataframe_by_author(training_features_dataframe)
    training_targets_groupby_author = get_targets_from_dataframe(training_features_dataframe_groupby_author, 
                                                                 training_targets_dataframe)

    # We choose to exclude outliers only on the basis of anomalous hourly activity
    outlier_strategies = {'activity' : custom_preprocessing.activity_based_outlier_rejection}
    
    final_features, final_targets = outlier_strategies['activity'](training_features_dataframe_groupby_author,
                                                                       training_targets_groupby_author)
    return final_features, final_targets

def get_test_data():
    test_features_dataframe = get_dataframes_from_csv("data/test_data.csv")
    test_features_dataframe_groupby_author = group_dataframe_by_author(test_features_dataframe)
    return test_features_dataframe_groupby_author

#######################
# Persist GridCV data #
#######################

def save_dict_to_disk(results : dict):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    pd.DataFrame.from_dict(results).to_csv(f'output/gridsearch_{timestamp}.csv') 
    
#########################
# Persist Final Results #
#########################

def save_submission_to_disk(test_data, predictions):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    submission = pd.DataFrame({"author" : test_data.author, "gender" : predictions})
    submission.to_csv(f'submissions/predictions_{timestamp}.csv', index=False)      
        
        
