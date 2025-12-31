import torch
from torch.utils.data import DataLoader, TensorDataset
from amr_utility import get_seq_label_simple
from imblearn.over_sampling import RandomOverSampler, SMOTE


def seq_to_num(seq):
    swap = {'A': 0.25, 'G': 0.5, 'C': 0.75, 'T': 1.0}
    sequences = [[swap[char] for char in s] for s in seq]
    return sequences

def seq_to_one_hot(seq):
    swap = {'A': [1.0, 0.0, 0.0, 0.0], 'G': [0.0, 1.0, 0.0, 0.0], 'C': [0.0, 0.0, 1.0, 0.0], 'T': [0.0, 0.0, 0.0, 1.0]}
    sequences = [[num for char in s for num in swap[char]] for s in seq]
    return sequences


def ros_resample(X, y):
    ros = RandomOverSampler(sampling_strategy=1.0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    return X_resampled, y_resampled


def smote_resample(X, y):
    smote = SMOTE(sampling_strategy=1.0)
    X_smote, y_smote = smote.fit_resample(X, y)
    return X_smote, y_smote


def get_seq_datasets(dataset="Staphylococcus_aureus_cefoxitin_pbp4"):
    seq_data = get_seq_label_simple(dataset)
    seq_train = [x[0] for x in seq_data["train"]]
    y_train = [x[1] for x in seq_data["train"]]

    seq_test = [x[0] for x in seq_data["test"]]
    y_test = [x[1] for x in seq_data["test"]]

    seq_test = seq_to_one_hot(seq_test)
    seq_train = seq_to_one_hot(seq_train)

    print(len(y_train) - sum(y_train))
    print(len(seq_train))

    # seq_train, y_train = ros_resample(seq_train, y_train)
    #print(len(seq_train))
    # print(y_train)
    # print(len(y_train) - sum(y_train))

    train = TensorDataset(torch.tensor(seq_train), torch.tensor(y_train))
    test = TensorDataset(torch.tensor(seq_test), torch.tensor(y_test))

    return train, test


def get_dataloader(dataset="Staphylococcus_aureus_cefoxitin_pbp4", batch_size=32, shuffle=True):
    train, test = get_seq_datasets(dataset)
    return (DataLoader(train, batch_size=batch_size, shuffle=shuffle),
            DataLoader(test, batch_size=batch_size, shuffle=False))
