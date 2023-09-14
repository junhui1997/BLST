import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split



def butter_lowpass_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def moving_average(data, window_size):
    window = np.ones(window_size) / float(window_size)
    return np.convolve(data, window, 'same')


def apply_filter(df, args):
    if args.filter == 'low_pass':
        cutoff = 0.1
        fs = 10
        for idx in range(len(df.columns)):
            if df[df.columns[idx]].dtype == 'float64':
                filtered_data = butter_lowpass_filter(df[df.columns[idx]], cutoff, fs, order=2)
                df[df.columns[idx]] = filtered_data
    if args.filter == 'mean':
        for idx in range(len(df.columns)):
            if df[df.columns[idx]].dtype == 'float64':
                filtered_data = moving_average(df[df.columns[idx]], window_size=5)
                df[df.columns[idx]] = filtered_data
    return df

class Dataset_bone_drill_c(Dataset):
    def __init__(self, flag='train', args=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = args.seq_len
        self.enc_in = args.enc_in
        self.seed = args.seed

        # init
        # 对于classification任务只会划分为train test
        self.args = args
        assert flag in ['TRAIN', 'TEST']
        self.flag = flag
        self.__read_data__()

    def __read_data__(self):
        '''
          df_raw.columns: ['date', ...(other features), target feature]
        '''
        self.scaler = StandardScaler()
        drill_folder = '../dataset/'
        file_name = 'df_label.pkl'
        df = pd.read_pickle(drill_folder + file_name)
        df = df.drop(columns='time_stamp')
        self.scaler.fit(df.iloc[:, :self.enc_in].values)
        df.iloc[:, :self.enc_in] = self.scaler.transform(df.iloc[:, :self.enc_in].values)
        if self.args.filter != 'no_filter':
            df = apply_filter(df, self.args)

        val_list = []
        label_list = []
        file_list = []
        for i in range(self.seq_len, df.shape[0], self.seq_len):
            val = df.iloc[i - self.seq_len:i, :self.enc_in].to_numpy().astype('float64')
            label = df.iloc[i]['label']
            file_name = df.iloc[i]['file_name']
            val_list.append(val)
            label_list.append(label)
            file_list.append(file_name)
        df_clean = pd.DataFrame({"value list": val_list, "label": label_list, 'file_name': file_list})
        df_clean = df_clean[::2]  # classifcaion
        split_mode = 'file'
        # split_mode = 'single'
        if split_mode == 'single':
            df_clean = df_clean[::2]
            x_train, x_test = train_test_split(df_clean, test_size=0.2, random_state=self.seed)
        elif split_mode == 'file':
            k_fold = 4
            name_list = list(df_clean['file_name'].unique())
            if (self.seed+1)*k_fold >= len(name_list):
                end = -1
            else:
                end = (self.seed+1)*k_fold
            start = self.seed*k_fold
            train_list = name_list[0:start]+name_list[end::]
            test_list = name_list[start:end]
            x_train = df_clean[df_clean['file_name'].isin(train_list)]
            x_test = df_clean[df_clean['file_name'].isin(test_list)]
        x_train = x_train.reset_index(drop=True)
        x_test = x_test.reset_index(drop=True)
        ans = x_train['label'].isnull().any()
        ans_t = x_test['label'].isnull().any()
        y_train = x_train['label']
        y_test = x_test['label']
        if self.flag == 'TRAIN':
            self.data_x = x_train
            self.data_y = y_train
            self.ds_len = len(y_train)
        elif self.flag == 'TEST':
            self.data_x = x_test
            self.data_y = y_test
            self.ds_len = len(y_test)

    def __getitem__(self, index):
        data = self.data_x['value list'].iloc[index].astype('float64')
        data = torch.from_numpy(data)
        label = torch.tensor([self.data_y[index]]).to(torch.long)
        return data, label

    def __len__(self):
        return self.ds_len

