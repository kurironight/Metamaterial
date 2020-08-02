import random
import torch
from torch.utils.data import DataLoader
from .convert import convert_npy_to_torch


def shuffle_dataset(dataset, ratio=[8, 1, 1], seed=1):
    index = list(range(len(dataset)))
    random.seed(seed)  # static seed
    random.shuffle(index)
    one_length = int(len(dataset) / sum(ratio))
    train_index = torch.zeros(len(index), dtype=torch.bool)
    train_index[index[:one_length * ratio[0]]] = True
    train_dataset = dataset[train_index]
    val_index = torch.zeros(len(index), dtype=torch.bool)
    val_index[index[one_length*ratio[0]:one_length *
                    (ratio[0]+ratio[1])]] = True
    val_dataset = dataset[val_index]
    test_index = torch.zeros(len(index), dtype=torch.bool)
    test_index[index[one_length * (ratio[0]+ratio[1]):]] = True
    test_dataset = dataset[test_index]

    print(len(train_dataset), len(val_dataset), len(test_dataset))
    return train_dataset, val_dataset, test_dataset


def split_data_train_eval_test(data_dir, batch_size=50):
    structures, E_data, G_data = convert_npy_to_torch(data_dir)
    structures, E_data, G_data = structures.float(), E_data.float(), G_data.float()
    print("データの数: ", structures.shape[0])

    train_structures, eval_structures,\
        test_structures = shuffle_dataset(structures)
    train_E_data, eval_E_data,\
        test_E_data = shuffle_dataset(E_data)
    train_G_data, eval_G_data,\
        test_G_data = shuffle_dataset(G_data)
    train_loader = torch.utils.data.TensorDataset(
        train_structures, train_E_data, train_G_data)
    eval_loader = torch.utils.data.TensorDataset(
        eval_structures, eval_E_data, eval_G_data)
    test_loader = torch.utils.data.TensorDataset(
        test_structures, test_E_data, test_G_data)

    train_loader = DataLoader(
        train_loader, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(
        eval_loader, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_loader, batch_size=batch_size, shuffle=True)

    return train_loader, eval_loader, test_loader
