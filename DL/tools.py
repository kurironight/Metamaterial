import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from .convert import convert_npy_to_torch
from GA.fem import calc_E, calc_G
import os


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


def show_results(dl_model, pth_path, data_dir, save_dir=None, batch_size=50):
    model = dl_model()
    model_name = model.name
    cuda = torch.cuda.is_available()
    if cuda:
        print('cuda available!')
        if torch.cuda.device_count() > 1:
            model.load_state_dict(torch.load(pth_path))
            print('load for multi!')
            print('\nYou can use {} GPU'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model).cuda()
        else:
            print('\n you use only one GPU')
            model.cuda()
            model.load_state_dict(torch.load(pth_path))
    model.eval()
    _, _, test_loader = split_data_train_eval_test(
        data_dir, batch_size)
    outputs_E = []
    outputs_G = []
    E_list = []
    G_list = []

    with torch.no_grad():
        for _, [structure, E, G] in enumerate(test_loader):
            if (E.size()[0] != batch_size):
                break
            # 条件入力ベクトルを入れたい
            if cuda:
                structure, E, G = structure.cuda(), E.cuda(), G.cuda()
            output_structure = model(E, G)
            output_structure = torch.reshape(output_structure, (-1, 32, 32))
            Es = torch.reshape(E, (-1, 1))
            Gs = torch.reshape(G, (-1, 1))
            output_structure = output_structure.to(
                'cpu').detach().numpy().copy()
            Es = Es.to('cpu').detach().numpy().copy()
            Gs = Gs.to('cpu').detach().numpy().copy()

            for structure, E, G in zip(output_structure, Es, Gs):
                output_E = calc_E(structure)
                output_G = calc_G(structure)
                outputs_E.append(output_E)
                outputs_G.append(output_G)
                E_list.append(E)
                G_list.append(G)
            break

    if save_dir is None:
        dir_path = os.path.dirname(pth_path)
        save_dir = os.path.join(dir_path, "results")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "result.txt"), mode='a') as f:
            f.writelines("model_name:{}\n".format(model_name))
            f.writelines("data_path:{}\n".format(data_dir))
            f.writelines("pth_path:{}\n".format(pth_path))
            f.writelines("DL_model:{}\n".format(dl_model()))
            E_mean_error = np.mean(
                (np.abs(np.array(outputs_E)-np.array(E_list))/np.array(E_list)))
            E_max = np.max(np.abs(np.array(outputs_E) -
                                  np.array(E_list))/np.array(E_list))
            G_mean_error = np.mean(
                (np.abs(np.array(outputs_G)-np.array(G_list))/np.array(G_list)))
            G_max = np.max(np.abs(np.array(outputs_G) -
                                  np.array(G_list))/np.array(G_list))
            f.writelines("Error: E mean {} max {}\n".format(
                E_mean_error, E_max))
            f.writelines("Error: G mean {} max {}\n".format(
                G_mean_error, G_max))


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
