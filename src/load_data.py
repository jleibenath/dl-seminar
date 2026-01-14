import torch
from torch.utils.data import DataLoader, TensorDataset
from amr_utility import get_seq_label_simple

def seq_to_one_hot(seq):
    swap = {'A': [1.0, 0.0, 0.0, 0.0], 'G': [0.0, 1.0, 0.0, 0.0], 'C': [0.0, 0.0, 1.0, 0.0], 'T': [0.0, 0.0, 0.0, 1.0]}
    sequences = [[num for char in s for num in swap[char]] for s in seq]
    return sequences


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

    train = TensorDataset(torch.tensor(seq_train), torch.tensor(y_train))
    test = TensorDataset(torch.tensor(seq_test), torch.tensor(y_test))

    return train, test


def get_dataloader(dataset="Staphylococcus_aureus_cefoxitin_pbp4", batch_size=32, shuffle=True):
    train, test = get_seq_datasets(dataset)
    return (DataLoader(train, batch_size=batch_size, shuffle=shuffle),
            DataLoader(test, batch_size=batch_size, shuffle=False))
