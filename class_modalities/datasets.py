import numpy as np
import pandas as pd
import os
import glob

from sklearn.model_selection import train_test_split


class DataManager(object):

    def __init__(self, base_path=None, csv_path=None):
        self.base_path = base_path
        self.csv_path = csv_path
        self.seed = 42  # random state
        self.test_size = 0.15
        self.val_size = 0.15

    def get_data(self):

        PET_ids = np.sort(glob.glob(os.path.join(self.base_path, '*_nifti_PT.nii')))
        CT_ids = np.sort(glob.glob(os.path.join(self.base_path, '*_nifti_CT.nii')))
        MASK_ids = np.sort(glob.glob(os.path.join(self.base_path, '*_nifti_mask.nii')))
        return list(zip(PET_ids, CT_ids)), MASK_ids

    def get_subset(self, subset, wrap_with_dict=True):
        assert wrap_with_dict
        assert subset in ['train', 'val', 'test']

        df = pd.read_csv(self.csv_path)
        df = df[df['PET'] == 'pet0']  # select only pet 0 exam

        if 'subset' not in df.columns:
            key_split = 'PATIENT ID'  # unique id
            idx = np.arange(df[key_split].nunique())
            split = np.empty(df[key_split].nunique(), dtype="<U6")

            idx_train, idx_test = train_test_split(idx, test_size=self.test_size, random_state=self.seed)

            size = self.val_size / (1 - self.test_size)
            idx_train, idx_val = train_test_split(idx_train, test_size=size, random_state=self.seed)

            split[idx_train] = 'train'
            split[idx_val] = 'val'
            split[idx_test] = 'test'

            df_patient = pd.DataFrame(data={key_split: df[key_split].unique(),
                                            'subset': split})
            df = df.merge(df_patient, on=key_split, how='left')

        return self.wrap_in_list_of_dict(df[df['subset'] == subset])

    def get_train_val_test(self, wrap_with_dict=False):
        if self.csv_path is None:
            X, y = self.get_data()
            return self.split_train_val_test_split(X, y, random_state=self.seed)
        else:
            df = pd.read_csv(self.csv_path)
            df = df[df['PET'] == 'pet0']  # select only pet 0 exam

            if 'subset' not in df.columns:
                key_split = 'PATIENT ID'  # unique id
                idx = np.arange(df[key_split].nunique())
                split = np.empty(df[key_split].nunique(), dtype="<U6")

                idx_train, idx_test = train_test_split(idx, test_size=self.test_size, random_state=self.seed)

                size = self.val_size / (1 - self.test_size)
                idx_train, idx_val = train_test_split(idx_train, test_size=size, random_state=self.seed)

                split[idx_train] = 'train'
                split[idx_val] = 'val'
                split[idx_test] = 'test'

                df_patient = pd.DataFrame(data={key_split: df[key_split].unique(),
                                                'subset': split})
                df = df.merge(df_patient, on=key_split, how='left')

            df_train = df[df['subset'] == 'train']
            df_val = df[df['subset'] == 'val']
            df_test = df[df['subset'] == 'test']

            if wrap_with_dict:
                return self.wrap_in_list_of_dict(df_train), self.wrap_in_list_of_dict(
                    df_val), self.wrap_in_list_of_dict(df_test)
            else:
                X_train, y_train = list(zip(df_train['NIFTI_PET'].values, df_train['NIFTI_CT'].values)), df_train[
                    'NIFTI_MASK'].values
                X_val, y_val = list(zip(df_val['NIFTI_PET'].values, df_val['NIFTI_CT'].values)), df_val[
                    'NIFTI_MASK'].values
                X_test, y_test = list(zip(df_test['NIFTI_PET'].values, df_test['NIFTI_CT'].values)), df_test[
                    'NIFTI_MASK'].values

                return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def wrap_in_dict(df):
        """
        :return: {'pet_img': [pet_img0, pet_img1, ...],
                  'ct_img': [ct_img0, ct_img1, ..],
                  'mask_img': [mask_img0, mask_img1, ...]}
        """
        # df.T.to_dict().values()
        return {'pet_img': df['NIFTI_PET'].values,
                'ct_img': df['NIFTI_CT'].values,
                'mask_img': df['NIFTI_MASK'].values}

    @staticmethod
    def wrap_in_list_of_dict(df):
        """
        :return: [ {'pet_img': pet_img0, 'ct_img': ct_img0, 'mask_img': mask_img0},
                    {'pet_img': pet_img1, 'ct_img': ct_img1, 'mask_img': mask_img1},
                    {'pet_img': pet_img2, 'ct_img': ct_img2, 'mask_img': mask_img2}, ...]
        """
        # return df[['NIFTI_PET', 'NIFTI_CT', 'NIFTI_MASK']].T.to_dict().values()
        mapper = {'NIFTI_PET': 'pet_img', 'NIFTI_CT': 'ct_img', 'NIFTI_MASK': 'mask_img'}
        return df[['NIFTI_PET', 'NIFTI_CT', 'NIFTI_MASK']].rename(columns=mapper).to_dict('records')

    @staticmethod
    def split_train_val_test_split(X, y, test_size=0.15, val_size=0.15, random_state=42):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=size, random_state=random_state)

        return X_train, X_val, X_test, y_train, y_val, y_test
