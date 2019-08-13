#!/home/brian/miniconda3/bin/python3.7
# encoding: utf-8
"""
Read the docs, obey PEP 8 and PEP 20 (Zen of Python, import this)

Build on:    Spyder
Python vver: 3.7.3

Created on Tue Jul 16 17:03:30 2019

@author: brian
"""

# Standord packages:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Use the above notation for all !!!!!
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import GridSearchCV

from itertools import product
from itertools import combinations, permutations, combinations_with_replacement

import sys
import multiprocessing as mp

# Imports from own modules
from make_data import make_data


# ***** Start here *****
pool = mp.Pool(processes=(mp.cpu_count() - 1))
skf = StratifiedKFold(n_splits=5, random_state=42)
scaler = StandardScaler()
np.random.seed(42)

# %% Import transformed Data


X_train, y_train, X_test, y_test = make_data()


# %% Playing with ColumnTransformer and PipeLines


class ModelTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.features = None

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        if (X.shape[0] > 500) and (X.shape[0] < 2000):
            if X.shape[0] < 1500:
                y = y_train[train_index]
            else:
                y = y_train
            skf = StratifiedKFold(n_splits=5, random_state=42)
            dfx = pd.DataFrame(index=[*range(X.shape[0])], columns=[self.name])
            for train_ind, val_ind in skf.split(X, y):
                X_tra = X[train_ind]
                X_val = X[val_ind]
                y_tra = y[train_ind]

                self.model.fit(X_tra, y_tra)
                dfx.loc[val_ind, self.name] = self.model.predict(X_val)
            self.model.fit(X, y)
            self.features = pd.get_dummies(dfx).columns.tolist()
            return pd.get_dummies(dfx)

        else:
            dd = pd.get_dummies(pd.DataFrame(self.model.predict(X),
                                             columns=[self.name]))
            dd = dd.reindex(columns=self.features).fillna(0)
            return dd


def run_clf(param):
    global train_index
    clf.set_params(**param)
    total = list()
    skf = StratifiedKFold(n_splits=4, random_state=42)
    for train_index, val_index in skf.split(X_train, y_train):
        X_tra = X_train[train_index]
        X_val = X_train[val_index]
        y_tra = y_train[train_index]
        y_val = y_train[val_index]

        clf.fit(X_tra, y_tra)

        total.append(clf.score(X_val, y_val))

    return (np.mean(total), param)


# %%Setting up the pipeline with columntransformer

spec_transform = FeatureUnion([
        ('knn', ModelTransformer(KNeighborsClassifier(), 'knn')),
        ('dtc', ModelTransformer(DecisionTreeClassifier(), 'dtc')),
        ('rbs', ModelTransformer(SVC(kernel='rbf'), 'rbs')),
        ])

chro_transform = FeatureUnion([
        ('pol', ModelTransformer(SVC(kernel='poly'), 'pol')),
        ('lin', ModelTransformer(SVC(kernel='linear'), 'lin')),
        ('rbc', ModelTransformer(SVC(kernel='rbf'), 'rbc'))
        ])

base_transform = ColumnTransformer(transformers=[
        ('spec', spec_transform, [*range(36)]),
        ('chro', chro_transform, slice(36, 192))
        ])

est_log = Pipeline([('LOG', LogisticRegression())])
est_rfc = Pipeline([('RFC', RandomForestClassifier())])
est_knn = Pipeline([('KNN', KNeighborsClassifier())])

clf = Pipeline(steps=[('base_predictions', base_transform),
                      ('scaler', StandardScaler),
                      ('meta_predictions', est_log)
                      ])


# %% Homemade Gradient booster, more like random search:
def ransea():
    param_grid_values = {
        'base_predictions__spec__knn__model__n_neighbors': [5] +
        [1, 2, 3, 4, 5],
        'base_predictions__spec__knn__model__weights': ['uniform'] +
        ['uniform', 'distance'],
        'base_predictions__spec__knn__model__p': [2] +
        [1, 2],

        'base_predictions__spec__dtc__model__criterion': ['gini'] +
        ['gini', 'entropy'],
        'base_predictions__spec__dtc__model__max_depth': [None] +
        [*range(10, 151, 20)],
        'base_predictions__spec__dtc__model__random_state': [42],

        'base_predictions__spec__rbs__model__gamma': [0.005] +
        [*np.linspace(0.001, 0.007, 7)],
        'base_predictions__spec__rbs__model__C': [1] +
        [0.3, 0.5, 0.7, 1, 1.1, 1.5],
        'base_predictions__spec__rbs__model__coef0': [0],
        'base_predictions__spec__rbs__model__random_state': [42],

        'base_predictions__chro__pol__model__gamma': [0.005] +
        [*np.linspace(0.001, 0.007, 7)],
        'base_predictions__chro__pol__model__degree': [4] +
        [5, 6, 7, 8, 9, 10],
        'base_predictions__chro__pol__model__C': [1] +
        [*np.linspace(0.7, 1.6, 10)],
        'base_predictions__chro__pol__model__coef0': [0],
        'base_predictions__chro__pol__model__random_state': [42],

        'base_predictions__chro__lin__model__C': [1] +
        [0.1, 0.2, 0.4, 0.7, 1, 1.1, 1.5],
        'base_predictions__chro__lin__model__coef0': [0],
        'base_predictions__chro__lin__model__random_state': [42],

        'base_predictions__chro__rbc__model__gamma': [0.005] +
        [*np.linspace(0.001, 0.007, 7)],
        'base_predictions__chro__rbc__model__C': [1] +
        [0.05, 0.1, 0.2, 0.4, 0.7, 1],
        'base_predictions__chro__rbc__model__coef0': [0],
        'base_predictions__chro__rbc__model__random_state': [42],

        'meta_predictions__LOG__penalty': ['l2'],
        'meta_predictions__LOG__C': [0.1] +
        [0.1, 0.2, 0.4, 0.5],
        'meta_predictions__LOG__solver': ['lbfgs'],
        'meta_predictions__LOG__multi_class': ['multinomial'],
        'meta_predictions__LOG__max_iter': [1000],
        'meta_predictions__LOG__random_state': [42],
        }

    top_score = 0.8613
    search_params = [k for k, value in
                     param_grid_values.items() if len(value) > 1]

    run_dict = dict()
    for key in search_params:
        run_dict[key] = [param_grid_values[key][0]]

    proba_count = np.array([1]*len(search_params))

    param_grid_basic = dict()
    for key, value in param_grid_values.items():
        if key in search_params:
            # nr = np.random.choice(len(value)-1)
            # param_grid_basic[key] = [value[nr], value[nr+1]]
            param_grid_basic[key] = [value[1], value[-1]]   #
        else:
            param_grid_basic[key] = value

    for i in range(8):

        part_param = np.random.choice(search_params, size=4,
                                      replace=False,
                                      p=proba_count/np.sum(proba_count))

        param_grid = param_grid_basic.copy()
        for key, value in param_grid_basic.items():
            if key not in search_params:
                param_grid[key] = value
            elif key not in part_param:
                param_grid[key] = [value[0]]

        pgpg = list(ParameterGrid(param_grid))
        print(f'Combinations: {len(pgpg)}')

        with mp.Pool() as pool:
            results = pool.map(run_clf, pgpg)

        best_score = sorted(results, key=lambda x: x[0])[-1][0]
        print(f'Score: {best_score}')
        print()

        if top_score > best_score:
            continue

        top_score = best_score

        comb = []
        for t in results:
            if t[0] == best_score:
                comb.append(t[1])

        for key in part_param:
            scor = np.sum([1 if t[key] == param_grid[key][0] else -1 for
                           t in comb])
            param_list = param_grid_values[key]
            ind_0 = param_list.index(param_grid[key][0])
            ind_1 = param_list.index(param_grid[key][1])

            if scor < 0:
                if param_grid_values[key][0] > param_grid[key][1]:
                    param_grid_values[key][0] = param_grid[key][1]
                if ind_0 == ind_1 - 1:
                    ind_0 = 1
                else:
                    ind_0 += 1
            elif scor > 0:
                if param_grid_values[key][0] < param_grid[key][0]:
                    param_grid_values[key][0] = param_grid[key][0]

                if ind_1 == ind_0 + 1:
                    ind_1 = len(param_list) - 1
                else:
                    ind_1 -= 1
            else:
                temp = param_grid_values[key][0]
                if temp < param_grid[key][0]:
                    param_grid_values[key][0] = param_grid[key][0]
                elif temp > param_grid[key][1]:
                    param_grid_values[key][0] = param_grid[key][1]

            param_grid_basic[key] = [param_grid_values[key][ind_0],
                                     param_grid_values[key][ind_1]]

            run_dict[key].append(param_grid_basic[key])

    for key, value in run_dict.items():
        print(key)
        print(value)
    print()
    for key, value in param_grid_basic.items():
        print(key)
        print(value[0])


ransea()


# %% Baseline:
param_grid_default = {
        'base_predictions__spec__knn__model__n_neighbors': [1],
        'base_predictions__spec__knn__model__weights': ['uniform'],
        'base_predictions__spec__knn__model__p': [1],

        'base_predictions__spec__dtc__model__criterion': ['gini'],
        'base_predictions__spec__dtc__model__max_depth': [60],
        'base_predictions__spec__dtc__model__random_state': [42],

        'base_predictions__spec__rbs__model__gamma': ['scale'],
        'base_predictions__spec__rbs__model__C': [0.3],
        'base_predictions__spec__rbs__model__coef0': [0],
        'base_predictions__spec__rbs__model__random_state': [42],

        'base_predictions__chro__pol__model__gamma': ['scale'],
        'base_predictions__chro__pol__model__degree': [5],
        'base_predictions__chro__pol__model__C': [0.7],
        'base_predictions__chro__pol__model__coef0': [0],
        'base_predictions__chro__pol__model__random_state': [42],

        'base_predictions__chro__lin__model__gamma': ['scale'],
        'base_predictions__chro__lin__model__C': [0.1],
        'base_predictions__chro__lin__model__coef0': [0],
        'base_predictions__chro__lin__model__random_state': [42],

        'base_predictions__chro__rbc__model__gamma': ['scale'],
        'base_predictions__chro__rbc__model__C': [0.05],
        'base_predictions__chro__rbc__model__coef0': [0],
        'base_predictions__chro__rbc__model__random_state': [42],

        'meta_predictions__LOG__penalty': ['l2'],
        'meta_predictions__LOG__C': [0.1],
        'meta_predictions__LOG__solver': ['lbfgs'],
        'meta_predictions__LOG__multi_class': ['multinomial'],
        'meta_predictions__LOG__max_iter': [1000],
        'meta_predictions__LOG__random_state': [42],
    }

pgpg = list(ParameterGrid(param_grid_default))
print(f'Combinations: {len(pgpg)}')

with mp.Pool() as pool:
    results = pool.map(run_clf, pgpg)

best_score = sorted(results, key=lambda x: x[0])[-1][0]
print(f'Score: {best_score}')
print()
# Score 0.853409

# %% Making final run
param_grid_fit = dict([(k, v[0]) for k, v in param_grid_default.items()])
clf.set_params(**param_grid_fit)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
# 0.970408 on default settings


# %% Standard values:
"""

param_grid_log = {
    'base_predictions__spec__knn__model__n_neighbors': [5],
    'base_predictions__spec__knn__model__weights': ['uniform'],
    'base_predictions__spec__knn__model__p': [2],

    'base_predictions__spec__dtc__model__criterion': ['gini'],
    'base_predictions__spec__dtc__model__max_depth': [None],
    'base_predictions__spec__dtc__model__random_state': [42],

    'base_predictions__spec__rbs__model__gamma': ['scale'],
    'base_predictions__spec__rbs__model__C': [1],
    'base_predictions__spec__rbs__model__coef0': [0],
    'base_predictions__spec__rbs__model__random_state': [42],

    'base_predictions__chro__pol__model__gamma': ['scale'],
    'base_predictions__chro__pol__model__degree': [4],
    'base_predictions__chro__pol__model__C': [1],
    'base_predictions__chro__pol__model__coef0': [0],
    'base_predictions__chro__pol__model__random_state': [42],

    'base_predictions__chro__lin__model__gamma': ['scale'],
    'base_predictions__chro__lin__model__C': [1],
    'base_predictions__chro__lin__model__coef0': [0],
    'base_predictions__chro__lin__model__random_state': [42],

    'base_predictions__chro__rbc__model__gamma': ['scale'],
    'base_predictions__chro__rbc__model__C': [1],
    'base_predictions__chro__lin__model__coef0': [0],
    'base_predictions__chro__rbc__model__random_state': [42],

    'meta_predictions__LOG__penalty': ['l2'],
    'meta_predictions__LOG__C': [1],
    'meta_predictions__LOG__solver': ['lbfgs'],
    'meta_predictions__LOG__multi_class': ['multinomial'],
    'meta_predictions__LOG__max_iter': [1000],
    'meta_predictions__LOG__random_state': [42],
    }
"""
