import random
import torch
import numpy as np
import matplotlib.pyplot as plt


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


def plot_history(history, save_path, save_title="learning curve"):
    epochs = history['epoch']
    train_loss, eval_loss, test_loss = history['train_loss'],\
        history['eval_loss'], history['test_loss']
    epochs = np.array(epochs)
    train_loss = np.array(train_loss)
    eval_loss = np.array(eval_loss)
    test_loss = np.array(test_loss)

    print('minimum of testloss: ' + str(min(test_loss)))
    print('min epoch is: ' + str(epochs[np.argmin(test_loss)]))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, train_loss, label='train_loss')
    ax.plot(epochs, eval_loss, label='eval_loss')
    ax.plot(epochs, test_loss, label='test_loss')
    ax.set_xlim(1, max(epochs))
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend()
    ax.set_title(save_title)
    plt.savefig(save_path)
