import os
import sys
import warnings

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.augmentation import run_augmentation_single
from utils.timefeatures import time_features

warnings.filterwarnings('ignore')



class Dataset_hour(Dataset):
    """
    Custom PyTorch Dataset for handling the hourly-level ETT (Electricity Transformer Temperature) dataset.
    """

    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args

        # Set default sizes if not provided
        if size is None:
            self.seq_len = 24 * 4 * 4  # Sequence length of input
            self.label_len = 24 * 4  # Length of the labels
            self.pred_len = 24 * 4  # Prediction length
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # Validate the flag type
        assert flag in ['train', 'test', 'val'], "flag must be one of ['train', 'test', 'val']"
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        # Read the dataset
        self.__read_data__()

    def __read_data__(self):
        """
        Read and preprocess data, including scaling and generating timestamp features.
        """
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Set data boundaries for train, validation, and test sets
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Select columns based on feature type
        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Scale data if needed
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Generate time features
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].apply(lambda row: row.month)
            df_stamp['day'] = df_stamp['date'].apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp['date'].apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp['date'].apply(lambda row: row.hour)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        # Data augmentation for training
        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, _ = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """
        Get individual data sample by index.
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """
        Apply inverse transformation to the scaled data.
        """
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_minute(Dataset):
    """
    Custom PyTorch Dataset for handling the minute-level ETT dataset.
    """

    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args

        # Set default sizes if not provided
        if size is None:
            self.seq_len = 24 * 4 * 4  # Sequence length of input
            self.label_len = 24 * 4  # Length of the labels
            self.pred_len = 24 * 4  # Prediction length
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # Validate the flag type
        assert flag in ['train', 'test', 'val'], "flag must be one of ['train', 'test', 'val']"
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        # Read the dataset
        self.__read_data__()

    def __read_data__(self):
        """
        Read and preprocess data, including scaling and generating timestamp features.
        """
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Set data boundaries for train, validation, and test sets
        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.args.valid_date is None:
            num_rows = 473422
            num_rows = df_raw.shape[0]
            break_point_idx0 = (int(num_rows * 0.7) // (30 * 24 * 4)) * 30 * 24 * 4
            break_point_idx1 = (int(num_rows * 0.85) // (30 * 24 * 4)) * 30 * 24 * 4
        else:
            diff = (pd.to_datetime(df_raw['date']) - self.args.valid_date).abs()
            valid_idx = diff.idxmin()
            closest_date = df_raw.loc[valid_idx, 'date']
            if self.set_type == 2:
                print(f'closest_date: {closest_date}')
            num_rows = df_raw.shape[0]
            break_point_idx0 = (int(valid_idx * 0.8) // (30 * 24 * 4)) * 30 * 24 * 4
            break_point_idx1 = (valid_idx // (30 * 24 * 4)) * 30 * 24 * 4

        border1s = [0, break_point_idx0 - self.seq_len, break_point_idx1 - self.seq_len]
        border2s = [break_point_idx0, break_point_idx1, (num_rows// (30 * 24 * 4)) * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Select columns based on feature type
        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Scale data if needed
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Generate time features
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].apply(lambda row: row.month)
            df_stamp['day'] = df_stamp['date'].apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp['date'].apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp['date'].apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp['date'].apply(
                lambda row: row.minute // 15)  # Aggregating minutes into 15-minute intervals
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        # Data augmentation for training
        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, _ = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """
        Get individual data sample by index.
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """
        Apply inverse transformation to the scaled data.
        """
        return self.scaler.inverse_transform(data)