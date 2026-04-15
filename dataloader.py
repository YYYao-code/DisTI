import os
import pickle

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from itertools import groupby
from operator import itemgetter


from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class GeneralLoader(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = torch.FloatTensor(self.x[index])
        y = torch.FloatTensor(self.y[index])
        return x, y


class TrainingLoader(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = torch.FloatTensor(self.x[index])
        return x


def normalization(x):
    min_value = x.min()
    max_value = x.max()
    return (x - min_value) / (max_value - min_value)


# Generated fast fourier transformed sequences.
def torch_fft_trasnform(seq):
    torch_seq = torch.from_numpy(seq)
    # freq length
    tp_cnt = seq.shape[1]
    tm_period = seq.shape[1]

    # FFT
    ft_ = torch.fft.fft(torch_seq, dim=1) / tm_period
    # Half
    ft_ = ft_[:, range(int(tm_period / 2)), :]
    # index
    val_ = np.arange(int(tp_cnt / 2))
    # freq axis
    freq = val_ / tm_period

    ffts_tensor = abs(ft_)
    ffts = ffts_tensor.numpy()
    return ffts, freq


# Generated training sequences for use in the model.
def _create_sequences(values, seq_length, stride, historical=False):

    seq = []
    if historical:
        for i in range(seq_length, len(values) + 1, stride):
            seq.append(values[i - seq_length:i])
    else:
        for i in range(0, len(values) - seq_length + 1, stride):
            seq.append(values[i: i + seq_length])


    return np.stack(seq)


def _count_anomaly_segments(values):
    values = np.where(values == 1)[0]
    anomaly_segments = []

    for k, g in groupby(enumerate(values), lambda ix: ix[0] - ix[1]):
        anomaly_segments.append(list(map(itemgetter(1), g)))
    return len(anomaly_segments), anomaly_segments



def load_PSM(seq_length=100, stride=1, historical=False):

    path = './datasets/PSM/'

    x_train, x_valid, x_test = [], [], []
    y_valid, y_test = [], []
    y_segment_valid, y_segment_test = [], []
    train_seq, label_seq, test_seq = [], [], []

    train_df = pd.read_csv(f'{path}/train.csv').iloc[:, 1:].fillna(method="ffill").values
    test_df = pd.read_csv(f'{path}/test.csv').iloc[:, 1:].fillna(method="ffill").values
    labels = pd.read_csv(f'{path}/test_label.csv')['label'].values.astype(int)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df = scaler.fit_transform(train_df)
    test_df = scaler.transform(test_df)

    valid_idx = int(test_df.shape[0] * 0.3)
    valid_df = test_df[:valid_idx]
    valid_labels = labels[:valid_idx]

    if seq_length > 0:
        x_train.append(_create_sequences(train_df, seq_length, stride, historical))
        x_valid.append(_create_sequences(valid_df, seq_length, stride, historical))
        x_test.append(_create_sequences(test_df, seq_length, stride, historical))
        y_test.append(np.expand_dims(_create_sequences(labels, seq_length, stride), axis=-1))
        y_valid.append(np.expand_dims(_create_sequences(valid_labels, seq_length, stride), axis=-1))
    else:
        x_train.append(train_df)
        x_valid.append(valid_df)
        x_test.append(test_df)
        y_test.append(labels)

    label_seq.append(labels)
    test_seq.append(test_df)
    train_seq.append(train_df)

    y_segment_valid.append(_count_anomaly_segments(valid_labels)[1])
    y_segment_test.append(_count_anomaly_segments(label_seq)[1])

    return {'x_train': x_train, 'x_valid': x_valid, 'x_test': x_test,
            'label_seq': label_seq, 'test_seq': test_seq, 'train_seq': train_seq,
            'y_valid': y_valid, 'y_test': y_test,
            'y_segment_valid': y_segment_valid, 'y_segment_test': y_segment_test}


def load_SKAB(seq_length=100, stride=1, historical=False):

    path = './datasets/SKAB/data/'

    x_train, x_valid, x_test = [], [], []
    y_valid, y_test = [], []
    y_segment_valid, y_segment_test = [], []
    train_seq, label_seq, test_seq = [], [], []

    train_df = pickle.load(open(f'{path}/SKAB_train.pkl', 'rb'))
    test_df = pickle.load(open(f'{path}/SKAB_test.pkl', 'rb'))
    labels = pickle.load(open(f'{path}/SKAB_test_label.pkl', 'rb')).astype(int)


    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df = scaler.fit_transform(train_df)
    test_df = scaler.transform(test_df)

    valid_idx = int(test_df.shape[0] * 0.3)
    valid_df = test_df[:valid_idx]
    valid_labels = labels[:valid_idx]

    if seq_length > 0:
        x_train.append(_create_sequences(train_df, seq_length, stride, historical))
        x_valid.append(_create_sequences(valid_df, seq_length, stride, historical))
        x_test.append(_create_sequences(test_df, seq_length, stride, historical))
        y_test.append(np.expand_dims(_create_sequences(labels, seq_length, stride), axis=-1))
        y_valid.append(np.expand_dims(_create_sequences(valid_labels, seq_length, stride), axis=-1))
    else:
        x_train.append(train_df)
        x_valid.append(valid_df)
        x_test.append(test_df)
        y_test.append(labels)

    label_seq.append(labels)
    test_seq.append(test_df)
    train_seq.append(train_df)

    y_segment_valid.append(_count_anomaly_segments(valid_labels)[1])
    y_segment_test.append(_count_anomaly_segments(label_seq)[1])


    return {'x_train': x_train, 'x_valid': x_valid, 'x_test': x_test,
            'label_seq': label_seq, 'test_seq': test_seq, 'train_seq': train_seq,
            'y_valid': y_valid, 'y_test': y_test,
            'y_segment_valid': y_segment_valid, 'y_segment_test': y_segment_test}


def load_SMD(seq_length=100, stride=1, historical=False):

    path = f'./datasets/SMD'
    f_names = sorted([f for f in os.listdir(f'{path}/train') if os.path.isfile(os.path.join(f'{path}/train', f))])
    f_names = [f.split('_')[0] for f in f_names]

    x_train, x_valid, x_test = [], [], []
    y_valid, y_test = [], []
    y_segment_valid, y_segment_test = [], []
    label_seq, test_seq, train_seq = [], [], []

    for f_name in f_names:
        train_df = pd.read_pickle(f'{path}/train/{f_name}' + '_train.pkl')
        test_df = pd.read_pickle(f'{path}/test/{f_name}' + '_test.pkl')
        labels = pd.read_pickle(f'{path}/test_label/{f_name}' + '_test_label.pkl').astype(int)


        scaler = MinMaxScaler(feature_range=(0, 1))
        train_df = scaler.fit_transform(train_df)
        test_df = scaler.transform(test_df)

        valid_idx = int(test_df.shape[0] * 0.3)
        valid_df = test_df[:valid_idx]
        valid_labels = labels[:valid_idx]

        if seq_length > 0:
            x_train.append(_create_sequences(train_df, seq_length, stride, historical))
            x_valid.append(_create_sequences(valid_df, seq_length, stride, historical))
            x_test.append(_create_sequences(test_df, seq_length, stride, historical))
            y_test.append(np.expand_dims(_create_sequences(labels, seq_length, stride), axis=-1))
            y_valid.append(np.expand_dims(_create_sequences(valid_labels, seq_length, stride), axis=-1))
        else:
            x_train.append(train_df)
            x_valid.append(valid_df)
            x_test.append(test_df)
            y_test.append(labels)

        label_seq.append(labels)
        test_seq.append(test_df)
        train_seq.append(train_df)

        y_segment_valid.append(_count_anomaly_segments(valid_labels)[1])
        y_segment_test.append(_count_anomaly_segments(label_seq)[1])


    return {'x_train': x_train, 'x_valid': x_valid, 'x_test': x_test,
            'label_seq': label_seq, 'test_seq': test_seq, 'train_seq': train_seq,
            'y_valid': y_valid, 'y_test': y_test,
            'y_segment_valid': y_segment_valid, 'y_segment_test': y_segment_test}




############################### Get Time Window ###############################
# form参数对于NIPS数据集，传入的是self.form,指定异常的特征。对于其他数据集，传入的是self.data_num,即指定dataset中的entity number
def get_loader_segment(batch_size, seq_length, form, step, mode='train', dataset='NeurIPSTS'):
    if dataset == 'PSM':
        # Create sliding window sequences
        data_dict = load_PSM(seq_length=seq_length, stride=step)
        x_train, x_test, y_test = data_dict['x_train'][form], data_dict['x_test'][form], data_dict['y_test'][form]
        label_seq, test_seq = data_dict['label_seq'][form], data_dict['test_seq'][form]
        x_valid, y_valid = data_dict['x_valid'][form], data_dict['y_valid'][form]


    elif dataset == 'SKAB':
        data_dict = load_SKAB(seq_length=seq_length, stride=step)
        x_train, x_test, y_test = data_dict['x_train'][form], data_dict['x_test'][form], data_dict['y_test'][form]
        label_seq, test_seq = data_dict['label_seq'][form], data_dict['test_seq'][form]
        x_valid, y_valid = data_dict['x_valid'][form], data_dict['y_valid'][form]


    elif dataset == 'SMD':
        # Create sliding window sequences
        data_dict = load_SMD(seq_length=seq_length, stride=step)
        x_train, x_test, y_test = data_dict['x_train'][form], data_dict['x_test'][form], data_dict['y_test'][form]
        label_seq, test_seq = data_dict['label_seq'][form], data_dict['test_seq'][form]

        x_valid, y_valid = data_dict['x_valid'][form], data_dict['y_valid'][form]



        # Indexing
    if mode == 'train':
        dataset = TrainingLoader(x_train)
        shuffle = True
    elif mode == 'vali':
        dataset = GeneralLoader(x_valid, y_valid)
        shuffle = True
    else:
        dataset = GeneralLoader(x_test, y_test)
        shuffle = False
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader, label_seq, test_seq


