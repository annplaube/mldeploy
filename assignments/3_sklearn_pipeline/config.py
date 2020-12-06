# ====   PATHS ===================

TRAINING_DATA_FILE = "titanic.csv"
PIPELINE_NAME = 'logistic_regression.pkl'


# ======= FEATURE GROUPS =============

TARGET = 'survived'

INDICATE_MISSING_VARS = ['age', 'fare']

CATEGORICAL_VARS = ['sex', 'cabin', 'embarked', 'title']

NUMERICAL_VARS = ['pclass', 'age', 'sibsp', 'parch', 'fare']

CABIN = ['cabin']

FEATURES = [
    'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'cabin',
    'embarked', 'title'
]
