import numpy as np
import os
import matplotlib.pyplot as plt
import glob


def convert_folder_npy_to_image(folder_path):
    """フォルダ内のnpyファイル全てを画像化する関数

    Args:
        folder_path (str): フォルダのパス
    """
    npy_list = glob.glob(os.path.join(folder_path, "*.npy"))
    for PATH in npy_list:
        image = np.load(PATH)
        left_image = np.fliplr(image).copy()
        down_image = np.concatenate([left_image, image], 1)
        up_image = np.flipud(down_image).copy()
        whole_image = np.concatenate([up_image, down_image], 0)

        ny, nx = whole_image.shape
        x = np.arange(0, nx+1)  # x軸の描画範囲の生成。
        y = np.arange(0, ny+1)  # y軸の描画範囲の生成。
        X, Y = np.meshgrid(x, y)
        fig = plt.figure()
        _ = plt.pcolormesh(X, Y, whole_image, cmap="binary")
        plt.axis("off")
        dirname, basename = os.path.split(PATH)
        fig.savefig(os.path.join(dirname, "image_"+basename[:-4]+".png"))
        plt.close()


"""
convert_folder_npy_to_image(os.path.join(
    "data\\bar_nx_20_ny_20\\gen_10_pa_10"))
"""
