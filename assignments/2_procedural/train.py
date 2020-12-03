import preprocessing_functions as pf
import config

# ================================================
# TRAINING STEP - IMPORTANT TO PERPETUATE THE MODEL

# Load data
df = pf.load_data(config.PATH_TO_DATASET)

# divide data set
X_train, X_test, y_train, y_test = pf.divide_train_test(df, config.TARGET)

# get first letter from cabin variable
pf.extract_cabin_letter(X_train, 'cabin')

# impute numerical variable
for var in config.NUMERICAL_TO_IMPUTE:
    pf.add_missing_indicator(X_train, var)
    pf.impute_na(X_train, var, config.IMPUTATION_DICT[var])

# impute categorical variables
for var in config.CATEGORICAL_VARS:
    pf.impute_na(X_train, var)

# Group rare labels
for var, labels in config.FREQUENT_LABELS.items():
    pf.remove_rare_labels(X_train, var, labels)

# encode categorical variables
for var in config.CATEGORICAL_VARS:
    X_train = pf.encode_categorical(X_train, var)

# check all dummies were added
pf.check_dummy_variables(X_train, config.DUMMY_VARIABLES)

X_train = X_train[config.ALL_VARS]

# train scaler and save
pf.train_scaler(X_train, config.OUTPUT_SCALER_PATH)

# scale train set
X_train = pf.scale_features(X_train, config.OUTPUT_SCALER_PATH)

# train model and save
pf.train_model(X_train, y_train, config.OUTPUT_MODEL_PATH)


print('Finished training')