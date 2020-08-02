import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from GA.fem import calc_E, calc_G
import os
import pickle
from .tools import split_data_train_eval_test
from GA.tools import make_structure_from_numpy
from tqdm import tqdm


def save_test_results(dl_model, pth_path, data_dir, save_dir=None, data_num=None):
    """学習済みpthとDLモデルクラス，学習データディレクトリを指定し，テストデータにおける結果を表示する

    Args:
        dl_model (object): DLモデルのクラス
        pth_path (str): 学習したpthのパスを指定
        data_dir (str)): データのディレクトリ指定
        save_dir (str), optional): 保存するディレクトリを指定できる.指定しなければ，pth_pathのディレクトリ上に保存される.
        data_num (int), optional): 用いるテストデータの数を指定できる Defaults to None.
    """
    model = dl_model()
    model_name = model.name
    cuda = torch.cuda.is_available()
    if cuda:
        if torch.cuda.device_count() > 1:
            model.load_state_dict(torch.load(pth_path))
            model = nn.DataParallel(model).cuda()
        else:
            model.cuda()
            model.load_state_dict(torch.load(pth_path))
    E_list, G_list, outputs_E_list, outputs_G_list, output_structure_list,\
        structure_list = make_result_list(
            model, data_dir, cuda, data_num=data_num)

    # save the results
    if save_dir is None:
        dir_path = os.path.dirname(pth_path)
        save_dir = os.path.join(dir_path, "results")
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'E_list.pkl'), 'wb') as f:
        pickle.dump(E_list, f)
    with open(os.path.join(save_dir, 'G_list.pkl'), 'wb') as f:
        pickle.dump(G_list, f)
    with open(os.path.join(save_dir, 'outputs_E_list.pkl'), 'wb') as f:
        pickle.dump(outputs_E_list, f)
    with open(os.path.join(save_dir, 'outputs_G_list.pkl'), 'wb') as f:
        pickle.dump(outputs_G_list, f)
    with open(os.path.join(save_dir, 'output_structure_list.pkl'), 'wb') as f:
        pickle.dump(output_structure_list, f)
    with open(os.path.join(save_dir, 'structure_list.pkl'), 'wb') as f:
        pickle.dump(structure_list, f)

    with open(os.path.join(save_dir, "result.txt"), mode='a') as f:
        f.writelines("model_name:{}\n".format(model_name))
        f.writelines("data_path:{}\n".format(data_dir))
        f.writelines("pth_path:{}\n".format(pth_path))
        f.writelines("DL_model:{}\n".format(dl_model()))
        E_mean_error = np.mean(
            (np.abs(np.array(outputs_E_list)-np.array(E_list))/np.array(E_list)))
        E_max = np.max(np.abs(np.array(outputs_E_list) -
                              np.array(E_list))/np.array(E_list))
        G_mean_error = np.mean(
            (np.abs(np.array(outputs_G_list)-np.array(G_list))/np.array(G_list)))
        G_max = np.max(np.abs(np.array(outputs_G_list) -
                              np.array(G_list))/np.array(G_list))
        f.writelines("Error: E mean {} max {}\n".format(
            E_mean_error, E_max))
        f.writelines("Error: G mean {} max {}\n".format(
            G_mean_error, G_max))
    show_structure_image(save_dir)


def make_result_list(model, data_dir, cuda, batch_size=50, data_num=None):
    print("データリスト作成開始")
    model.eval()
    _, _, test_loader = split_data_train_eval_test(
        data_dir, batch_size)
    outputs_E_list = []
    outputs_G_list = []
    E_list = []
    G_list = []
    output_structure_list = []
    structure_list = []

    with torch.no_grad():
        data_total_num = 0
        flag = False
        for _, [structures, Es, Gs] in enumerate(tqdm(test_loader)):
            if (Es.size()[0] != batch_size):
                break
            # 条件入力ベクトルを入れたい
            if cuda:
                Es, Gs = Es.cuda(), Gs.cuda()
            output_structures = model(Es, Gs)
            output_structures = torch.reshape(output_structures, (-1, 32, 32))
            structures = torch.reshape(structures, (-1, 32, 32))
            Es = torch.reshape(Es, (-1, 1))
            Gs = torch.reshape(Gs, (-1, 1))
            output_structures = output_structures.to(
                'cpu').detach().numpy().copy()
            structures = structures.to('cpu').detach().numpy().copy()
            Es = Es.to('cpu').detach().numpy().copy()
            Gs = Gs.to('cpu').detach().numpy().copy()

            for structure, output_structure, E, G in zip(structures,
                                                         output_structures, Es, Gs):
                data_total_num += 1
                output_E = calc_E(output_structure)
                output_G = calc_G(output_structure)
                outputs_E_list.append(output_E)
                outputs_G_list.append(output_G)
                E_list.append(E)
                G_list.append(G)
                output_structure_list.append(output_structure)
                structure_list.append(structure)
                if (data_num is not None) and (data_total_num >= data_num):
                    flag = True
                    break
            if flag:
                break
    return E_list, G_list, outputs_E_list, outputs_G_list,\
        output_structure_list, structure_list


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
    plt.close()


def show_structure_image(data_dir):
    structure_path = os.path.join(data_dir, 'structure_list.pkl')
    output_structure_path = os.path.join(data_dir, 'output_structure_list.pkl')
    assert os.path.isfile(structure_path), structure_path+"が存在しない"
    assert os.path.isfile(
        output_structure_path), output_structure_path+"が存在しない"
    with open(os.path.join(structure_path), 'rb') as f:
        structures = pickle.load(f)
    with open(os.path.join(output_structure_path), 'rb') as f:
        output_structures = pickle.load(f)

    os.makedirs(os.path.join(data_dir, "real"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "output"), exist_ok=True)

    for num, (structure, output_structure) in enumerate(zip(structures,
                                                            output_structures)):
        num += 1
        make_structure_from_numpy(
            structure, os.path.join(data_dir, "real", "{}.png".format(num)))
        make_structure_from_numpy(
            output_structure, os.path.join(data_dir, "output", "{}.png".format(num)))
