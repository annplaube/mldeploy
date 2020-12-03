import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pickle import dump, load

import joblib


# Individual pre-processing and training functions
# ================================================

def load_data(df_path):
    # Function loads data for training
    df = pd.read_csv(df_path)
    return df


def divide_train_test(df, target):
    # Function divides data set in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(target, axis=1),
        df[target],
        test_size=0.2,
        random_state=0
    )
    return X_train, X_test, y_train, y_test


def extract_cabin_letter(df, var):
    # captures the first letter
    df[var] = df[var].str[0]


def add_missing_indicator(df, var):
    # function adds a binary missing value indicator
    df[var+'_NA'] = np.where(df[var].isnull(), 1, 0)


def impute_na(df, var, val='Missing'):
    # function replaces NA by value entered by user
    # or by string Missing (default behaviour)
    df[var].fillna(val, inplace=True)


def find_frequent_labels(df, var, rare_perc):
    df = df.copy()
    tmp = df.groupby(var)[var].count() / len(df)
    return tmp[tmp > rare_perc].index


def remove_rare_labels(df, var, frequent_ls):
    # groups labels that are not in the frequent list into the umbrella
    # group Rare

    df[var] = np.where(
        df[var].isin(frequent_ls), df[var], 'Rare'
    )


def encode_categorical(df, var):
    # adds ohe variables and removes original categorical variable

    df = df.copy()
    new_df = pd.concat(
        [df, pd.get_dummies(df[var], prefix=var, drop_first=True)],
        axis=1
    )
    new_df.drop(labels=var, axis=1, inplace=True)
    return new_df


def check_dummy_variables(df, dummy_list):
    # check that all missing variables where added when encoding, otherwise
    # add the ones that are missing
    for var in dummy_list:
        if not var in df.columns:
            df[var] = 0


def train_scaler(df, output_path):
    # train and save scaler
    scaler = StandardScaler()
    scaler.fit(df)
    joblib.dump(scaler, output_path)
    return scaler


def scale_features(df, output_path):
    # load scaler and transform data
    scaler = joblib.load(output_path)
    df = scaler.transform(df)
    return df


def train_model(df, target, output_path):
    # train and save model
    model = LogisticRegression(C=0.0005, random_state=0)
    model.fit(df, target)
    joblib.dump(model, output_path)


def predict(df, model):
    model = joblib.load(model)
    return model.predict(df)
