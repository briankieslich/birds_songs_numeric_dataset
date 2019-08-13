#!/home/brian/miniconda3/bin/python3.7
# encoding: utf-8
"""
Read the docs, obey PEP 8 and PEP 20 (Zen of Python, import this)

Build on:    Spyder
Python vver: 3.7.3

Created on Tue Aug 13 09:16:54 2019

@author: brian
"""

# Standord packages:
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


def make_data():
    data_train = pd.read_csv('data/train.csv')
    data_test = pd.read_csv('data/test.csv')

    data_train['g_s'] = data_train.genus + '_' + data_train.species
    data_test['g_s'] = data_test.genus + '_' + data_test.species

    y_train = data_train.g_s.values
    y_test = data_test.g_s.values

    features = list(data_train.columns)

    # Change column names
    spec_cols = [c for c in data_train.columns if c.startswith('spec_')]
    df_spec_train = data_train[spec_cols]
    df_spec_test = data_test[spec_cols]

    df_spec = pd.concat([df_spec_train, df_spec_test],
                        ignore_index=True, axis=0)

    columns = dict([(x, f'spec_{int(x.split("_")[2]):02}') for x in spec_cols])
    df_spec.rename(columns=columns, inplace=True)

    df_spec = df_spec[sorted(df_spec.columns)]

    df_velocity = pd.DataFrame(df_spec.loc[:, 'spec_01': 'spec_12'].values -
                               df_spec.loc[:, 'spec_00': 'spec_11'].values,
                               columns=[f'vel_{i:02}' for i in range(12)])
    df_acceleration = pd.DataFrame(
            df_velocity.loc[:, 'vel_01': 'vel_11'].values -
            df_velocity.loc[:, 'vel_00': 'vel_10'].values,
            columns=[f'acc_{i:02}' for i in range(11)])

    df_spec = pd.concat([df_spec, df_velocity, df_acceleration], axis=1)

    df_spec_train = df_spec.iloc[:1760].copy()
    df_spec_test = df_spec.iloc[1760:].copy().reset_index(drop=True)
    x_spec_train = scaler.fit_transform(df_spec_train)
    x_spec_test = scaler.transform(df_spec_test)

    chro_cols = [x for x in features if x.startswith('c')]
    x_chro_train = scaler.fit_transform(data_train[chro_cols])
    x_chro_test = scaler.transform(data_test[chro_cols])

    X_train = np.concatenate((x_spec_train, x_chro_train), axis=1)
    X_test = np.concatenate((x_spec_test, x_chro_test), axis=1)

    return X_train, y_train, X_test, y_test
