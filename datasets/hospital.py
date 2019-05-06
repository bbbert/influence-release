from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import numpy as np
import pandas as pd
import zipfile

from datasets.common import get_dataset_dir, maybe_download, DataSet
from tensorflow.contrib.learn.python.learn.datasets import base

def load_hospital(data_dir=None):
    dataset_dir = get_dataset_dir('hospital', data_dir=data_dir)
    hospital_path = os.path.join(dataset_dir, 'hospital.npz')

    if not os.path.exists(hospital_path) or True:
        HOSPITAL_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"

        raw_path = maybe_download(HOSPITAL_URL, 'dataset_diabetes.zip', dataset_dir)

        with zipfile.ZipFile(raw_path, 'r') as zipf:
            zipf.extractall(path=dataset_dir)

        csv_path = os.path.join(dataset_dir, 'dataset_diabetes', 'diabetic_data.csv')
        df = pd.read_csv(csv_path)

        # Convert categorical variables into numeric ones

        rng = np.random.RandomState(2)
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
        num_examples = len(Y)
        assert X.shape[0] == num_examples
        num_train_examples = 20000
        num_train_examples_per_class = int(num_train_examples / 2)
        num_test_examples = num_examples - num_train_examples
        assert num_test_examples > 0

        pos_idx = np.where(Y == 1)[0]
        neg_idx = np.where(Y == -1)[0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        assert len(pos_idx) + len(neg_idx) == num_examples

        train_idx = np.concatenate((pos_idx[:num_train_examples_per_class], neg_idx[:num_train_examples_per_class]))
        test_idx = np.concatenate((pos_idx[num_train_examples_per_class:], neg_idx[num_train_examples_per_class:]))
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

        X_train = np.array(X.iloc[train_idx, :], dtype=np.float32)
        Y_train = Y[train_idx]

        X_test = np.array(X.iloc[test_idx, :], dtype=np.float32)
        Y_test = Y[test_idx]

        lr_Y_train = np.array((Y_train + 1) / 2, dtype=int)
        lr_Y_test = np.array((Y_test + 1) / 2, dtype=int)

        #test_children_idx = np.where(X_test[:, age_var_indices[0]] == 1)[0]

        np.savez(hospital_path,
                 X_train=X_train,
                 Y_train=Y_train,
                 lr_Y_train=lr_Y_train,
                 X_test=X_test,
                 Y_test=Y_test,
                 lr_Y_test=lr_Y_test,
                 feature_names=feature_names)
    else:
        data = np.load(hospital_path)
        X_train = data['X_train']
        Y_train = data['Y_train']
        lr_Y_train = data['lr_Y_train']
        X_test = data['X_test']
        Y_test = data['Y_test']
        lr_Y_test = data['lr_Y_test']
        feature_names = data['feature_names']

    train = DataSet(X_train, Y_train, feature_names=feature_names)
    validation = None
    test = DataSet(X_test, Y_test, feature_names=feature_names)
    data_sets = base.Datasets(train=train, validation=validation, test=test)

    lr_train = DataSet(X_train, lr_Y_train, feature_names=feature_names)
    lr_validation = None
    lr_test = DataSet(X_test, lr_Y_test, feature_names=feature_names)
    lr_data_sets = base.Datasets(train=lr_train, validation=lr_validation, test=lr_test)

    return lr_data_sets

def load_hospital_preprocess(data_dir=None):
    dataset_dir = get_dataset_dir('hospital', data_dir=data_dir)
    hospital_path = os.path.join(dataset_dir, 'hospital_preprocess.npz')

    if not os.path.exists(hospital_path) or True:
        HOSPITAL_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"

        raw_path = maybe_download(HOSPITAL_URL, 'dataset_diabetes.zip', dataset_dir)

        with zipfile.ZipFile(raw_path, 'r') as zipf:
            zipf.extractall(path=dataset_dir)

        csv_path = os.path.join(dataset_dir, 'dataset_diabetes', 'diabetic_data.csv')
        df = pd.read_csv(csv_path)

        df = df.drop(['weight', 'payer_code', 'medical_specialty', 'citoglipton', 'examide'], axis=1)

        all_three_diagnoses_missing = (df['diag_1'] == '?') & (df['diag_2'] == '?') & (df['diag_3'] == '?')
        unknown_gender = df['gender'] == 'Unknown/Invalid'
        deceased = df['discharge_disposition_id'] == 11
        drop = all_three_diagnoses_missing | unknown_gender | deceased
        df = df[~drop]

        df['service_utilization'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']

        drugs = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'insulin', 'glyburide-metformin', 'tolazamide', 'metformin-pioglitazone','metformin-rosiglitazone', 'glimepiride-pioglitazone', 'glipizide-metformin', 'troglitazone', 'tolbutamide', 'acetohexamide']
        for drug in drugs:
            df['{}_change'.format(drug)] = ((df[drug] == 'Up') | (df[drug] == 'Down')).astype('int')
        df['med_uses'] = sum((df[drug] != 'No') for drug in drugs)
        df['med_changes'] = sum((df[drug] == 'Up') | (df[drug] == 'Down') for drug in drugs)

        df['level1_diag_1'] = df['diag_1']

        df.loc[df['diag_1'].str.contains('V'), 'level1_diag_1'] = 0
        df.loc[df['diag_1'].str.contains('E'), 'level1_diag_1'] = 0

        df['level1_diag_1'] = df['level1_diag_1'].replace('?', -1)

        icd = np.floor(df['level1_diag_1'].astype(float))

        df['level1_diag_1'] = 0
        df.loc[((icd >= 390) & (icd < 460)) | (icd == 785), 'level1_diag_1'] = 1
        df.loc[((icd >= 460) & (icd < 520)) | (icd == 786), 'level1_diag_1'] = 2
        df.loc[((icd >= 520) & (icd < 580)) | (icd == 787), 'level1_diag_1'] = 3
        df.loc[icd == 250, 'level1_diag_1'] = 4
        df.loc[((icd >= 800) & (icd < 1000)), 'level1_diag_1'] = 5
        df.loc[((icd >= 710) & (icd < 740)), 'level1_diag_1'] = 6
        df.loc[((icd >= 580) & (icd < 630)) | (icd == 788), 'level1_diag_1'] = 7
        df.loc[((icd >= 140) & (icd < 240)), 'level1_diag_1'] = 8

        df['level1_diag_1'] = df['level1_diag_1'].astype(int)

        df['admission_type_id'] = df['admission_type_id'].replace(2,1)
        df['admission_type_id'] = df['admission_type_id'].replace(7,1)
        df['admission_type_id'] = df['admission_type_id'].replace(6,5)
        df['admission_type_id'] = df['admission_type_id'].replace(8,5)

        df['change'] = (df['change'] == 'Ch').astype(int)
        df['gender'] = (df['gender'] == 'Male').astype(int)
        df['diabetesMed'] = (df['diabetesMed'] == 'Yes').astype(int)

        df[drugs] = (df[drugs] != 'No').astype(float)

        df['A1Cresult'] = df['A1Cresult'].replace('>7', 1)
        df['A1Cresult'] = df['A1Cresult'].replace('>8', 1)
        df['A1Cresult'] = df['A1Cresult'].replace('Norm', 0)
        df['A1Cresult'] = df['A1Cresult'].replace('None', -99)

        df['max_glu_serum'] = df['max_glu_serum'].replace('>200', 1)
        df['max_glu_serum'] = df['max_glu_serum'].replace('>300', 1)
        df['max_glu_serum'] = df['max_glu_serum'].replace('Norm', 0)
        df['max_glu_serum'] = df['max_glu_serum'].replace('None', -99)

        age_buckets = {
            '[{}-{})'.format(i, i + 10): i + 5
            for i in range(0, 100, 10)
        }
        df['age'] = df.age.map(age_buckets).astype('int64')

        df = df.drop_duplicates(subset=['patient_nbr'], keep='first')

        for race_value in np.unique(df['race'].values):
            df['race_{}'.format(race_value)] = (df['race'] == race_value).astype(int)
        del df['race']

        del df['diag_1']
        del df['diag_2']
        del df['diag_3']

        # Convert categorical variables into numeric ones
        rng = np.random.RandomState(2)
        X = pd.DataFrame()

        # Numerical variables that we can pull directly
        features = list(df.columns)
        features.remove('readmitted')
        X = df.loc[:, features].astype(float)

        print(X.shape)

        ### Set the Y labels
        readmitted = pd.Categorical(df.readmitted)

        Y = np.copy(readmitted.codes)

        # Combine >30 and 0 and flip labels, so 1 (>30) and 2 (No) become -1, while 0 becomes 1
        Y[Y >= 1] = -1
        Y[Y == 0] = 1

        # Map to feature names
        feature_names = X.columns.values

        ### Split into training and test sets.
        # For convenience, we balance the training set to have 10k positives and 10k negatives.
        num_examples = len(Y)
        assert X.shape[0] == num_examples
        num_train_examples = 20000
        num_train_examples_per_class = int(num_train_examples / 2)
        num_test_examples = num_examples - num_train_examples
        assert num_test_examples > 0

        pos_idx = np.where(Y == 1)[0]
        neg_idx = np.where(Y == -1)[0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        print(Y.shape)

        print(len(pos_idx))
        print(len(neg_idx))
        print(pos_idx[:20])
        assert len(pos_idx) + len(neg_idx) == num_examples

        train_idx = np.concatenate((pos_idx[:num_train_examples_per_class], neg_idx[:num_train_examples_per_class]))
        test_idx = np.concatenate((pos_idx[num_train_examples_per_class:], neg_idx[num_train_examples_per_class:]))
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
        print(len(train_idx))
        print(len(test_idx))

        X_train = np.array(X.iloc[train_idx, :], dtype=np.float32)
        Y_train = Y[train_idx]

        X_test = np.array(X.iloc[test_idx, :], dtype=np.float32)
        Y_test = Y[test_idx]

        print(X_train.shape, Y_train.shape)
        print(X_test.shape, Y_test.shape)

        lr_Y_train = np.array((Y_train + 1) / 2, dtype=int)
        lr_Y_test = np.array((Y_test + 1) / 2, dtype=int)

        #test_children_idx = np.where(X_test[:, age_var_indices[0]] == 1)[0]

        np.savez(hospital_path,
                 X_train=X_train,
                 Y_train=Y_train,
                 lr_Y_train=lr_Y_train,
                 X_test=X_test,
                 Y_test=Y_test,
                 lr_Y_test=lr_Y_test,
                 feature_names=feature_names)
    else:
        data = np.load(hospital_path)
        X_train = data['X_train']
        Y_train = data['Y_train']
        lr_Y_train = data['lr_Y_train']
        X_test = data['X_test']
        Y_test = data['Y_test']
        lr_Y_test = data['lr_Y_test']
        feature_names = data['feature_names']

    train = DataSet(X_train, Y_train, feature_names=feature_names)
    validation = None
    test = DataSet(X_test, Y_test, feature_names=feature_names)
    data_sets = base.Datasets(train=train, validation=validation, test=test)

    lr_train = DataSet(X_train, lr_Y_train)
    lr_validation = None
    lr_test = DataSet(X_test, lr_Y_test)
    lr_data_sets = base.Datasets(train=lr_train, validation=lr_validation, test=lr_test)

    return lr_data_sets
