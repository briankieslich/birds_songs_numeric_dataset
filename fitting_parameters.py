#!/home/brian/miniconda3/bin/python3.7
# encoding: utf-8
"""
Read the docs, obey PEP 8 and PEP 20 (Zen of Python, import this)

Build on:    Spyder
Python vver: 3.7.3

Created on Tue Aug 13 00:21:22 2019

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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import type_of_target

from itertools import product
from itertools import combinations, permutations, combinations_with_replacement
import time
import sys
import multiprocessing as mp


# ***** Start here *****
pool = mp.Pool(processes=(mp.cpu_count() - 1))
skf = StratifiedKFold(n_splits=5, random_state=42)
scaler = StandardScaler()
np.random.seed(42)
# %% Import data and make transformations

data_train = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')

data_train['g_s'] = data_train.genus + '_' + data_train.species
data_test['g_s'] = data_test.genus + '_' + data_test.species

y_train = data_train.g_s.values
y_test = data_test.g_s.values

# Detect genus only:
# y_train_all = data_train.genus.values
# y_test_all = data_test.genus.values


#%% Change column names
spec_cols = [c for c in data_train.columns if c.startswith('spec_')]
df_spec_train = data_train[spec_cols]
df_spec_test = data_test[spec_cols]

df_spec = pd.concat([df_spec_train, df_spec_test], ignore_index=True, axis=0)

columns=dict([(x, f'spec_{int(x.split("_")[2]):02}') for x in spec_cols])
df_spec.rename(columns=columns, inplace=True)

df_spec = df_spec[sorted(df_spec.columns)]


# Play en song
# %% Build some more features
df_velocity = pd.DataFrame(df_spec.loc[:, 'spec_01': 'spec_12'].values - 
                           df_spec.loc[:, 'spec_00': 'spec_11'].values, 
                           columns=[f'vel_{i:02}' for i in range(12)]) 
df_acceleration = pd.DataFrame(df_velocity.loc[:, 'vel_01': 'vel_11'].values -
                               df_velocity.loc[:, 'vel_00': 'vel_10'].values,
                               columns=[f'acc_{i:02}' for i in range(11)])

df_spec = pd.concat([df_spec, df_velocity, df_acceleration], axis=1)

df_spec_train = df_spec.iloc[:1760].copy()
df_spec_test = df_spec.iloc[1760:].copy().reset_index(drop=True)
x_spec_train = scaler.fit_transform(df_spec_train)
x_spec_test = scaler.transform(df_spec_test)

# Something about feature importance to show the above is a good idea.


# pca = PCA(n_components=14, random_state=42)
# x_spec_train = pca.fit(x_spec_train, y_train_all).transform(x_spec_train)
# x_spec_test = pca.transform(x_spec_test)

# %% and now to chro
chro_cols = [x for x in data_train.columns if x.startswith('c')]
x_chro_train = scaler.fit_transform(data_train[chro_cols])
x_chro_test = scaler.transform(data_test[chro_cols])

# pca = PCA(n_components=104, random_state=42)
# x_chro_train = pca.fit(x_chro_train, y_train_all).transform(x_chro_train)
# x_chro_test = pca.transform(x_chro_test)


# All together now. Why?? Pipeline !!
X_train = np.concatenate((x_spec_train, x_chro_train), axis=1)
X_test = np.concatenate((x_spec_test, x_chro_test), axis=1)



# %% Baseline with RandomForesrClassifier
#best = list()
#for n in range(138, 149, 1):
#    for m in range(55, 66, 1):
#        rfc = RandomForestClassifier(n_estimators=n, criterion='gini', max_depth=m)
#        rfc.fit(X_train, y_train)
#        best.append((
#                precision_score(y_test, rfc.predict(X_test), average='micro'),
#                n, m))
#sorted(best)[-1]


# 0.977986 n = 134, m = 37 on spec
# 0.976001 n = 144, m = 58 on chro
# 0.980873 n = 142, m = 57 on all


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
            for train_ind, val_ind in skf.split(X, y):# here we go
                X_tra = X[train_ind]
                X_val = X[val_ind]
                y_tra = y[train_ind]

                self.model.fit(X_tra, y_tra)
                dfx.loc[val_ind, self.name] = self.model.predict(X_val)
            self.model.fit(X, y)
            self.features = pd.get_dummies(dfx).columns.tolist()
            return pd.get_dummies(dfx)

        else:
            dd = pd.get_dummies(pd.DataFrame(self.model.predict(X), columns=[self.name]))
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
                      ('meta_predictions', est_log)
                      ])


# clf = Pipeline(steps=[('base_predictions', base_transform),
#                       ('meta_predictions', VotingClassifier(
#                               estimators=[
#                                       ('LOG', est_log),
#                                       ('RFC', est_rfc),
#                                       ('KNN', est_knn)
#                                       ],
#                               voting='soft'))
#                       ])


# %% Parameter grid for the Pipeline

# Parameters of pipelines can be set using ‘__’ separated parameter names:
param_grid_log = {
    'base_predictions__spec__knn__model__n_neighbors': [1], #
    'base_predictions__spec__knn__model__weights': ['distance'],
    'base_predictions__spec__knn__model__p': [2],  # 2

    'base_predictions__spec__dtc__model__criterion': ['gini'], # finale
    'base_predictions__spec__dtc__model__max_depth': [None],      
    'base_predictions__spec__dtc__model__random_state': [42],

    'base_predictions__spec__rbs__model__gamma': ['scale'], #
    'base_predictions__spec__rbs__model__C': [1.5], # *np.linspace(1, 1.3, 4)
    'base_predictions__spec__rbs__model__random_state': [42],

    'base_predictions__chro__pol__model__gamma': ['scale'], #
    'base_predictions__chro__pol__model__degree': [4], # final #final
    'base_predictions__chro__pol__model__C': [1], # 1 *np.linspace(0.8, 1.05, 6)
    'base_predictions__chro__pol__model__coef0': [0.10], # final
    'base_predictions__chro__pol__model__random_state': [42],

    'base_predictions__chro__lin__model__decision_function_shape': ['ovo'],
    'base_predictions__chro__lin__model__C': [1], #[*np.linspace(0.5, 1.5, 3)] + [0.1, 3, 7]
    'base_predictions__chro__lin__model__random_state': [42],

    'base_predictions__chro__rbc__model__gamma': ['scale'], #
    'base_predictions__chro__rbc__model__C': [0.375], #[*np.linspace(0.35, 0.39, 5)], # 0.36 < x,
    'base_predictions__chro__rbc__model__random_state': [42],

    'meta_predictions__LOG__penalty': ['l2'],
    'meta_predictions__LOG__C': [1], # 0.9
    'meta_predictions__LOG__solver': ['lbfgs'],
    'meta_predictions__LOG__multi_class': ['multinomial'],
    'meta_predictions__LOG__max_iter': [1000],
    'meta_predictions__LOG__random_state': [42],
    }

param_grid_rfc = {
    'base_predictions__spec__knn__model__n_neighbors': [4], # *range(4, 9, 1)
    'base_predictions__spec__knn__model__weights': ['distance'], # 'uniform' #
    'base_predictions__spec__knn__model__p': [2],  # final

    'base_predictions__spec__dtc__model__criterion': ['entropy'], #
    'base_predictions__spec__dtc__model__max_depth': [None], # final
    'base_predictions__spec__dtc__model__random_state': [42],

    'base_predictions__spec__rbs__model__gamma': ['scale'], #
    'base_predictions__spec__rbs__model__C': [0.96], # final
    'base_predictions__spec__rbs__model__random_state': [42],

    'base_predictions__chro__pol__model__gamma': ['scale'], #
    'base_predictions__chro__pol__model__degree': [3], # final
    'base_predictions__chro__pol__model__C': [0.74], # final
    'base_predictions__chro__pol__model__coef0': [0.49], #[*np.linspace(0.048, 0.051, 4)], # down
    'base_predictions__chro__pol__model__random_state': [42],

    'base_predictions__chro__lin__model__decision_function_shape': ['ovo'],
    'base_predictions__chro__lin__model__C': [1], #
    'base_predictions__chro__lin__model__random_state': [42],

    'base_predictions__chro__rbc__model__gamma': [0.032], #
    'base_predictions__chro__rbc__model__C': [0.84], #*np.linspace(0.82, 0.86, 5)
    'base_predictions__chro__rbc__model__random_state': [42],

    'meta_predictions__RFC__n_estimators': [104],  #[*range(103, 106, 1)],# *range(101, 112, 1)
    'meta_predictions__RFC__criterion': ['gini'],
    'meta_predictions__RFC__max_depth': [100], #[*range(99, 103, 1)], #*range(92, 105, 1), 100, 102 #
    'meta_predictions__RFC__random_state': [42],
    }

param_grid_knn = {
    'base_predictions__spec__knn_s__model__n_neighbors': [9],
    'base_predictions__spec__knn_s__model__weights': ['distance'],
    'base_predictions__spec__knn_s__model__p': [1],

    'base_predictions__spec__pol_s__model__gamma': ['scale'], #  
    'base_predictions__spec__pol_s__model__degree': [7], # final
    'base_predictions__spec__pol_s__model__C': [*np.linspace(0.27, 0.39, 7)], #
    #'base_predictions__spec__pol_s__model__coef0': [*np.linspace(0, 1, 6)],  #
    'base_predictions__spec__pol_s__model__random_state': [42],

    'base_predictions__chro__knn_c__model__n_neighbors': [5], #
    'base_predictions__chro__knn_c__model__weights': ['distance'], # final
    'base_predictions__chro__knn_c__model__p': [1],  #

    'base_predictions__chro__pol_c__model__gamma': ['scale'],  
    'base_predictions__chro__pol_c__model__degree': [8],
    'base_predictions__chro__pol_c__model__C': [5], #
    #'base_predictions__chro__pol_c__model__coef0': [*np.linspace(0, 1, 6)], #  
    'base_predictions__chro__pol_c__model__random_state': [42],

    'meta_predictions__KNN__n_neighbors': [4, 5], #
    'meta_predictions__KNN__weights': ['uniform', 'distance'], #
    'meta_predictions__KNN__p': [1, 2], #
    }

param_grid_all = {
    'base_predictions__spec__knn__model__n_neighbors': [5], #
    'base_predictions__spec__knn__model__weights': ['uniform'], #
    'base_predictions__spec__knn__model__p': [2],  #

    'base_predictions__spec__dtc__model__criterion': ['gini'], #
    'base_predictions__spec__dtc__model__max_depth': [None],
    'base_predictions__spec__dtc__model__random_state': [42],
    
    'base_predictions__spec__rbs__model__gamma': ['scale'], #
    'base_predictions__spec__rbs__model__C': [1],  #*
    'base_predictions__spec__rbs__model__random_state': [42],

    'base_predictions__chro__pol__model__gamma': ['scale'], #
    'base_predictions__chro__pol__model__degree': [3], #
    'base_predictions__chro__pol__model__C': [1], #
    'base_predictions__chro__pol__model__coef0': [0], #
    'base_predictions__chro__pol__model__random_state': [42],

    'base_predictions__chro__lin__model__decision_function_shape': ['ovr'],
    'base_predictions__chro__lin__model__C': [1],#
    'base_predictions__chro__lin__model__random_state': [42],

    'base_predictions__chro__rbc__model__gamma': ['scale'], #
    'base_predictions__chro__rbc__model__C': [1], #
    'base_predictions__chro__rbc__model__random_state': [42],

    'meta_predictions__RFC__RFC__n_estimators': [100], #
    'meta_predictions__RFC__RFC__criterion': ['gini'],
    'meta_predictions__RFC__RFC__max_depth': [None], #
    'meta_predictions__RFC__RFC__random_state': [42],

    # 'meta_predictions__KNN__KNN__n_neighbors': [5], #
    # 'meta_predictions__KNN__KNN__weights': ['uniform'], #
    # 'meta_predictions__KNN__KNN__p': [2], #

    'meta_predictions__LOG__LOG__penalty': ['l2'],
    'meta_predictions__LOG__LOG__C': [1],
    'meta_predictions__LOG__LOG__solver': ['lbfgs'],
    'meta_predictions__LOG__LOG__multi_class': ['multinomial'],
    'meta_predictions__LOG__LOG__max_iter': [1000],
    'meta_predictions__LOG__LOG__random_state': [42],

    }

param_grid = param_grid_log
pgpg = list(ParameterGrid(param_grid))
print(f'Combinations: {len(pgpg)}')


# %% Homemade Gradient booster, more like random search:
def ransea():
    param_grid_values = {
        'base_predictions__spec__knn__model__n_neighbors': [5],  #
        'base_predictions__spec__knn__model__weights': ['uniform', 'distance'],
        'base_predictions__spec__knn__model__p': [2],  # 2

        'base_predictions__spec__dtc__model__criterion': ['gini'],  #
        'base_predictions__spec__dtc__model__max_depth': [*range(10, 41, 5)],
        'base_predictions__spec__dtc__model__random_state': [42],

        'base_predictions__spec__rbs__model__gamma': ['scale'],  #
        'base_predictions__spec__rbs__model__C': [*np.linspace(0.1, 1.1, 11)],
        'base_predictions__spec__rbs__model__random_state': [42],

        'base_predictions__chro__pol__model__gamma': ['scale'],  #
        'base_predictions__chro__pol__model__degree': [6, 7, 8, 9, 10],  #
        'base_predictions__chro__pol__model__C': [*np.linspace(0.4, 0.8, 21)],  #
        'base_predictions__chro__pol__model__coef0': [*np.linspace(0.04, 0.07, 16)], # final
        'base_predictions__chro__pol__model__random_state': [42],
    
        'base_predictions__chro__lin__model__decision_function_shape': ['ovo'],
        'base_predictions__chro__lin__model__C': [*np.linspace(0.1, 0.3, 11)], #
        'base_predictions__chro__lin__model__random_state': [42],
    
        'base_predictions__chro__rbc__model__gamma': ['scale'], #
        'base_predictions__chro__rbc__model__C': [*np.linspace(0.2, 1.2, 11)], #
        'base_predictions__chro__rbc__model__random_state': [42],
    
        'meta_predictions__LOG__penalty': ['l2'],
        'meta_predictions__LOG__C': [*np.linspace(0.4, 1.4, 11)], # 0.9
        'meta_predictions__LOG__solver': ['lbfgs'],
        'meta_predictions__LOG__multi_class': ['multinomial'],
        'meta_predictions__LOG__max_iter': [1000],
        'meta_predictions__LOG__random_state': [42],
        }


    search_params = [k for k, value in param_grid_values.items()
                                    if len(value) > 1]

    run_dict = dict()
    for key in search_params:
        run_dict[key] = []

    param_grid_basic = dict()
    for key, value in param_grid_values.items():
        if key in search_params:
            # nr = np.random.choice(len(value)-1)
            # param_grid_basic[key] = [value[nr], value[nr+1]]
            param_grid_basic[key] = [value[0], value[-1]]   #
        else:
            param_grid_basic[key] = value

    for i in range(24):
        number_run = np.array([1/(len(run_dict[key])+0.1) for
                               key in search_params])

        part_param = np.random.choice(search_params, size=6,
                                      replace=False,
                                      p=number_run/np.sum(number_run))

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
            ind_x = np.random.choice(len(param_list))

            if scor > 0:
                pass

            if scor < 0:
                ind_0, ind_1 = ind_1, ind_0
            if (ind_0 < ind_1):
                if ind_0 > 0:
                    ind_1 = ind_0 - 1
                else:
                    ind_1 = ind_x
            else:
                if ind_0 < len(param_list) - 1:
                    ind_1 = ind_0 + 1
                else:
                    ind_1 = ind_x

            param_grid_basic[key] = [param_grid_values[key][ind_0],
                        param_grid_values[key][ind_1]]

            run_dict[key].append(param_grid_values[key][ind_0])


    for key, value in run_dict.items():
        print(key)
        print(value)


ransea()


# %% Examine the grid
comb = []
param_keys = list()
for key, value in param_grid.items():
    if len(value) > 1:
        param_keys.append(key)

for t in results:
    if t[0] == best_score:
        temp = list()
        for key in param_keys:
            temp.append(t[1][key])
        comb.append(temp)

for s in param_keys:
    print(s)
for t in comb:
    for u in t:
        print(f'{u:8}', end=' ')
    print()


# %% Make some graphs:
def single_graph(nr):
    x = list()
    for t in comb:
        x.append(t[nr])
    plt.hist(x, bins=100, rwidth=0.7);
    plt.title(f'{param_keys[nr]}')
    plt.show()


def double_graph(nr0, nr1):
    x = list()
    y = list()
    for t in comb:
        x.append(t[nr0])
        y.append(t[nr1])
    plt.hist2d(x, y, bins=[30, 30], density=True);
    plt.title(f'{param_keys[nr0]} \n {param_keys[nr1]}')
    #plt.ylim(0.12, 0.13)
    plt.show()


single_graph(3)
double_graph(0, 1)


# %% Making final run
param_grid_fit = dict([(k, v[0]) for k, v in param_grid.items()])
clf.set_params(**param_grid_fit)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))



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
