import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


# Add binary variable to indicate missing values
class MissingIndicator(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            variables = [variables]
        self.variables = variables

    def fit(self, X, y=None):
        # to accommodate sklearn pipeline functionality
        return self

    def transform(self, X):
        # add indicator
        X = X.copy()
        for var in self.variables:
            X[var + '_NA'] = np.where(X[var].isnull(), 1, 0)
        return X


# categorical missing value imputer
class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            variables = [variables]
        self.variables = variables


    def fit(self, X, y=None):
        # we need the fit statement to accommodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].fillna('Missing')
        return X


# Numerical missing value imputer
class NumericalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            variables = [variables]
        self.variables = variables

    def fit(self, X, y=None):
        # persist median in a dictionary
        self.imputer_dict_ = {}
        for var in self.variables:
            self.imputer_dict_[var] = X[var].median()
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var].fillna(self.imputer_dict_[var], inplace=True)
        return X


# Extract first letter from string variable
class ExtractFirstLetter(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            variables = [variables]
        self.variables = variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].str[0]
        return X


# frequent label categorical encoder
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, tol=0.05, variables=None):
        if not isinstance(variables, list):
            variables = [variables]
        self.variables = variables
        self.tol = tol

    def fit(self, X, y=None):
        # persist frequent labels in dictionary
        self.encoder_dict_ = {}
        for var in self.variables:
            t = X[var].value_counts() / len(X)
            self.encoder_dict_[var] = t[t >= self.tol].index.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = np.where(X[var].isin(self.encoder_dict_[var]), X[var], 'Rare')
        return X


# string to numbers categorical encoder
class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            variables = [variables]
        self.variables = variables

    def fit(self, X, y=None):

        self.dummies = pd.get_dummies(X[self.variables], drop_first=True).columns
        # learn order of columns to ensure train and test set have same order before
        train = self._encode(X)
        self.order = train.columns

        return self

    def _encode(self, X):
        # encode labels
        X = X.copy()
        # get dummies
        for var in self.variables:
            X = pd.concat(
                [X, pd.get_dummies(X[var], prefix=var, drop_first=True)],
                axis=1
            )
        # drop original variables
        X.drop(labels=self.variables, axis=1, inplace=True)
        # add missing dummies if any
        for dummy in self.dummies:
            if dummy not in X.columns:
                X[dummy] = 0
        return X

    def transform(self, X):
        X = self._encode(X)
        return X[self.order]
