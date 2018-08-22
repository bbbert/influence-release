from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import pandas as pd

from tensorflow.contrib.learn.python.learn.datasets import base
from influence.dataset import DataSet

def load_hospital():

    df = pd.read_csv('data/diabetic_data.csv')

    # Convert categorical variables into numeric ones

    X = pd.DataFrame()

    # Numerical variables that we can pull directly
    X = df.loc[
        :, 
        [
            'time_in_hospital',
            'num_lab_procedures',
            'num_procedures',
            'num_medications',
            'number_outpatient',
            'number_emergency',
            'number_inpatient',
            'number_diagnoses'
        ]]

    categorical_var_names = [
        'gender',
        'race',
        'age', 
        'discharge_disposition_id',
        'max_glu_serum',
        'A1Cresult',
        'metformin',
        'repaglinide',
        'nateglinide',
        'chlorpropamide',
        'glimepiride',
        'acetohexamide',
        'glipizide',
        'glyburide',
        'tolbutamide',
        'pioglitazone',
        'rosiglitazone',
        'acarbose',
        'miglitol',
        'troglitazone',
        'tolazamide',
        'examide',
        'citoglipton',
        'insulin',
        'glyburide-metformin',
        'glipizide-metformin',
        'glimepiride-pioglitazone',
        'metformin-rosiglitazone',
        'metformin-pioglitazone',
        'change',
        'diabetesMed'
    ]
    for categorical_var_name in categorical_var_names:
        categorical_var = pd.Categorical(
            df.loc[:, categorical_var_name])

        # Just have one dummy variable if it's boolean
        if len(categorical_var.categories) == 2:
            drop_first = True
        else:
            drop_first = False

        dummies = pd.get_dummies(
            categorical_var, 
            prefix=categorical_var_name,
            drop_first=drop_first)

        X = pd.concat([X, dummies], axis=1)

    ### Set the Y labels
    readmitted = pd.Categorical(df.readmitted)

    Y = np.copy(readmitted.codes)

    # Combine >30 and 0 and flip labels, so 1 (>30) and 2 (No) become -1, while 0 becomes 1
    Y[Y >= 1] = -1
    Y[Y == 0] = 1

    # Map to feature names
    feature_names = X.columns.values

    ### Find indices of age features
    age_var = pd.Categorical(df.loc[:, 'age'])
    age_var_names = ['age_%s' % age_var_name for age_var_name in age_var.categories]    
    age_var_indices = []
    for age_var_name in age_var_names:
        age_var_indices.append(np.where(X.columns.values == age_var_name)[0][0])
    age_var_indices = np.array(age_var_indices, dtype=int)

    ### Split into training and test sets. 
    # For convenience, we balance the training set to have 10k positives and 10k negatives.

    np.random.seed(2)
    num_examples = len(Y)
    assert X.shape[0] == num_examples
    num_train_examples = 20000
    num_train_examples_per_class = int(num_train_examples / 2)
    num_test_examples = num_examples - num_train_examples
    assert num_test_examples > 0

    pos_idx = np.where(Y == 1)[0]
    neg_idx = np.where(Y == -1)[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    assert len(pos_idx) + len(neg_idx) == num_examples

    train_idx = np.concatenate((pos_idx[:num_train_examples_per_class], neg_idx[:num_train_examples_per_class]))
    test_idx = np.concatenate((pos_idx[num_train_examples_per_class:], neg_idx[num_train_examples_per_class:]))
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    X_train = np.array(X.iloc[train_idx, :], dtype=np.float32)
    Y_train = Y[train_idx]

    X_test = np.array(X.iloc[test_idx, :], dtype=np.float32)
    Y_test = Y[test_idx]

    train = DataSet(X_train, Y_train, 0, np.zeros(len(X_train), dtype=bool))
    validation = None
    test = DataSet(X_test, Y_test, 0, np.zeros(len(X_test), dtype=bool))
    data_sets = base.Datasets(train=train, validation=validation, test=test)

    lr_train = DataSet(X_train, np.array((Y_train + 1) / 2, dtype=int), 0, np.zeros(len(X_train), dtype=bool))
    lr_validation = None
    lr_test = DataSet(X_test, np.array((Y_test + 1) / 2, dtype=int), 0, np.zeros(len(X_test), dtype=bool))
    lr_data_sets = base.Datasets(train=lr_train, validation=lr_validation, test=lr_test)

    test_children_idx = np.where(X_test[:, age_var_indices[0]] == 1)[0]

    return lr_data_sets






















