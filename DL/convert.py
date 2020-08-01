import numpy as np
import glob
import os
import re
import torch


def convert_npy_to_torch(folder_path):
    npy_list = glob.glob(os.path.join(folder_path, "E_*_G_*.npy"))
    save_path = os.path.join(folder_path, "for_load")
    if os.path.exists(save_path) and \
            np.load(os.path.join(save_path, 'E_data.npy')).shape[0] == len(npy_list):
        # 既にファイル変換処理が行われたものがある場合
        print("データをロードする")
        structures = np.load(os.path.join(save_path, 'structures.npy'))
        E_data = np.load(os.path.join(save_path, 'E_data.npy'))
        G_data = np.load(os.path.join(save_path, 'G_data.npy'))
    else:
        print("データを新規変換する")
        # サンプル数*チャネル数*縦*横
        structures = np.empty([1, 1, 32, 32], dtype=np.float64)
        E_data = np.empty([1, 1, 1], dtype=np.float64)
        G_data = np.empty([1, 1, 1], dtype=np.float64)

        for npy_path in npy_list:
            structure = np.load(npy_path)
            structures = np.concatenate(
                [structures, structure[np.newaxis, np.newaxis, :, :]], axis=0)
            basename = os.path.basename(npy_path)

            chara_value = re.findall("E_(.*)_G_(.*).npy", basename)
            E, G = [np.array([i], dtype=np.float64) for i in chara_value[0]]
            E_data = np.concatenate(
                [E_data, E[np.newaxis, np.newaxis, :]], axis=0)
            G_data = np.concatenate(
                [G_data, G[np.newaxis, np.newaxis, :]], axis=0)

        structures = structures[1:]
        E_data = E_data[1:]
        G_data = G_data[1:]

        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, 'structures.npy'), structures)
        np.save(os.path.join(save_path, 'E_data.npy'), E_data)
        np.save(os.path.join(save_path, 'G_data.npy'), G_data)

    structures = torch.from_numpy(structures)
    E_data = torch.from_numpy(E_data)
    G_data = torch.from_numpy(G_data)

    return structures, E_data, G_data


convert_npy_to_torch("data/bar_nx_32_ny_32/gen_15_pa_5")
