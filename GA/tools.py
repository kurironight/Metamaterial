from platypus import NSGAII
from tqdm import tqdm
from platypus import RandomGenerator, TournamentSelector
import numpy as np
import os
import matplotlib.pyplot as plt
import glob


class Original_NSGAII(NSGAII):
    """世代数を条件として指定することが出来る"""

    def __init__(self, problem,
                 population_size=100,
                 generator=RandomGenerator(),
                 selector=TournamentSelector(2),
                 variator=None,
                 archive=None,
                 **kwargs):
        super(NSGAII, self).__init__(
            problem, population_size, generator, **kwargs)
        self.selector = selector
        self.variator = variator
        self.archive = archive
        self.generations = 0

    def run(self, generation, callback=None):
        for i in tqdm(range(generation)):
            self.generations += 1
            self.step()

            if callback is not None:
                callback(self)


def convert_folder_npy_to_image(folder_path):
    """フォルダ内のnpyファイル全てを画像化する関数

    Args:
        folder_path (str): フォルダのパス
    """
    npy_list = glob.glob(os.path.join(folder_path, "*.npy"))
    for PATH in npy_list:
        structure = np.load(PATH)
        dirname, basename = os.path.split(PATH)
        save_path = os.path.join(dirname, "image_"+basename[:-4]+".png")
        make_structure_from_numpy(structure, save_path)


def make_structure_from_numpy(structure, save_path):
    left_structure = np.fliplr(structure).copy()
    down_structure = np.concatenate([left_structure, structure], 1)
    up_structure = np.flipud(down_structure).copy()
    whole_structure = np.concatenate([up_structure, down_structure], 0)
    ny, nx = whole_structure.shape
    x = np.arange(0, nx+1)  # x軸の描画範囲の生成。
    y = np.arange(0, ny+1)  # y軸の描画範囲の生成。
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    _ = plt.pcolormesh(X, Y, whole_structure, cmap="binary")
    plt.axis("off")
    fig.savefig(save_path)
    plt.close()
