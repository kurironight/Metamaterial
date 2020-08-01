import random
import torch


def shuffle_dataset(dataset, ratio=[8, 1, 1], seed=1):
    index = list(range(len(dataset)))
    random.seed(seed)  # static seed
    random.shuffle(index)
    one_length = int(len(dataset) / sum(ratio))
    train_index = torch.zeros(len(index), dtype=bool)
    train_index[index[:one_length * ratio[0]]] = True
    train_dataset = dataset[train_index]
    val_index = torch.zeros(len(index), dtype=bool)
    val_index[index[one_length*ratio[0]:one_length *
                    (ratio[0]+ratio[1])]] = True
    val_dataset = dataset[val_index]
    test_index = torch.zeros(len(index), dtype=bool)
    test_index[index[one_length * (ratio[0]+ratio[1]):]] = True
    test_dataset = dataset[test_index]

    print(len(train_dataset), len(val_dataset), len(test_dataset))
    return train_dataset, val_dataset, test_dataset
