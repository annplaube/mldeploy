from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import preprocessors as pp
import config


titanic_pipe = Pipeline(
    # complete with the list of steps from the preprocessors file
    # and the list of variables from the config
    [
        (
            'extract_cabin_letter',
            pp.ExtractFirstLetter(variables=config.CABIN)
        ),
        (
            'missing_indicator',
            pp.MissingIndicator(variables=config.INDICATE_MISSING_VARS)
        ),
        (
            'impute_categorical',
            pp.CategoricalImputer(variables=config.CATEGORICAL_VARS)
        ),
        (
            'impute_numerical',
            pp.NumericalImputer(variables=config.NUMERICAL_VARS)
        ),
        (
            'encode_rare',
            pp.RareLabelCategoricalEncoder(variables=config.CATEGORICAL_VARS, tol=0.05)
        ),
        (
            'encode_categorical',
            pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)
        ),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(C=0.0005, random_state=0)),
    ]
    )