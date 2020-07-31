from platypus import NSGAII, Problem, nondominated, Integer, Real, \
    CompoundOperator, SBX, HUX, PM, BitFlip, PCX
import matplotlib.pyplot as plt
from FEM import calc_E, calc_G
from make_structure import make_bar_structure, make_6_bar_edges
import numpy as np
import os
from convert_npy_to_image import convert_folder_npy_to_image
import time


def bar_multi_GA(nx=20, ny=20, volume_frac=0.5, parent=400, generation=100,
                 path="data"):
    generation = parent*generation
    PATH = os.path.join(path, "bar_nx_{}_ny_{}".format(nx, ny),
                        "gen_{}_pa_{}_vf{}".format(generation, parent, volume_frac))
    os.makedirs(PATH, exist_ok=True)
    start = time.time()

    def objective(vars):
        y_1, y_2, y_3, x_4, nodes, widths = convert_var_to_arg(vars)
        edges = make_6_bar_edges(nx, ny, y_1, y_2, y_3, x_4, nodes, widths)
        rho = make_bar_structure(nx, ny, edges)
        volume = np.sum(rho)/(nx*ny)

        return [calc_E(rho), calc_G(rho)], [volume]

    def convert_var_to_arg(vars):
        y_1 = vars[0]
        y_2 = vars[1]
        y_3 = vars[2]
        x_4 = vars[3]
        node_y_indexes = vars[4: 4 + 6 * 3]
        node_x_indexes = vars[4 + 6 * 3: 4 + 6 * 3 * 2]
        nodes = np.stack([node_x_indexes, node_y_indexes], axis=1)
        widths = vars[4 + 6 * 3 * 2:]
        return y_1, y_2, y_3, x_4, nodes, widths

    # 2変数2目的の問題
    problem = Problem(4+6*3*2+6*4, 2, 1)
    # 最小化or最大化を設定
    problem.directions[:] = Problem.MAXIMIZE

    # 決定変数の範囲を設定
    x_index_const = Real(1, nx)  # x座標に関する制約
    y_index_const = Real(1, ny)  # y座標に関する制約
    bar_constraint = Real(0, ny/2)  # バーの幅に関する制約
    problem.types[0:3] = y_index_const
    problem.types[3] = x_index_const
    problem.types[4: 4 + 6 * 3] = y_index_const
    problem.types[4 + 6 * 3: 4 + 6 * 3 * 2] = x_index_const
    problem.types[4 + 6 * 3 * 2:] = bar_constraint

    problem.constraints[:] = "<="+str(volume_frac)
    problem.function = objective
    problem.directions[:] = Problem.MAXIMIZE

    def print_result(algorithm):
        nondominated_solutions = nondominated(algorithm.result)
        for solution in [s for s in nondominated_solutions if s.feasible]:
            vars_list = [problem.types[i].decode(
                solution.variables[i]) for i in range(problem.nvars)]
            y_1, y_2, y_3, x_4, nodes, widths = convert_var_to_arg(vars_list)
            edges = make_6_bar_edges(nx, ny, y_1, y_2, y_3, x_4, nodes, widths)
            image = make_bar_structure(nx, ny, edges)
            np.save(os.path.join(PATH, 'E_{}_G_{}.npy'.format(
                solution.objectives[0], solution.objectives[1])), image)

    algorithm = NSGAII(problem, population_size=parent,
                       variator=PCX())
    algorithm.run(generation, callback=print_result)

    # グラフを描画

    fig = plt.figure()
    plt.scatter([s.objectives[0] for s in algorithm.result],
                [s.objectives[1] for s in algorithm.result], c="blue", label="infeasible solution")

    plt.scatter([s.objectives[0] for s in algorithm.result if s.feasible],
                [s.objectives[1] for s in algorithm.result if s.feasible], c="red", label='feasible solution')

    # 非劣解をとりだす
    nondominated_solutions = nondominated(algorithm.result)
    plt.scatter([s.objectives[0] for s in nondominated_solutions if s.feasible],
                [s.objectives[1] for s in nondominated_solutions if s.feasible], c="green", label="pareto solution")
    plt.legend(loc='lower left')

    plt.xlabel("$E$")
    plt.ylabel("$G$")
    fig.savefig(os.path.join(PATH, "graph.png"))
    plt.close()

    for solution in [s for s in nondominated_solutions if s.feasible]:
        vars_list = [problem.types[i].decode(
            solution.variables[i]) for i in range(problem.nvars)]
        y_1, y_2, y_3, x_4, nodes, widths = convert_var_to_arg(vars_list)
        edges = make_6_bar_edges(nx, ny, y_1, y_2, y_3, x_4, nodes, widths)
        image = make_bar_structure(nx, ny, edges)
        np.save(os.path.join(PATH, 'E_{}_G_{}.npy'.format(
            solution.objectives[0], solution.objectives[1])), image)

    convert_folder_npy_to_image(PATH)

    elapsed_time = time.time() - start

    with open("time.txt", mode='a') as f:
        f.writelines("bar_nx_{}_ny_{}_gen_{}_pa_vf{}:{}sec\n".format(
            nx, ny, generation, parent, volume_frac, elapsed_time))


def grid_multi_GA(nx=20, ny=20, volume_frac=0.5, parent=400, generation=100, path="data"):
    PATH = os.path.join(path, "grid_nx_{}_ny_{}".format(nx, ny),
                        "gen_{}_pa_{}".format(generation, parent))
    os.makedirs(PATH, exist_ok=True)
    start = time.time()

    def objective(vars):
        rho = np.array(vars)
        rho = rho.reshape(ny, nx-1)
        rho = np.concatenate([rho, np.ones((ny, 1))], 1)
        volume = np.sum(rho)/(nx*ny)

        return [calc_E(rho), calc_G(rho)], [volume]

    # 2変数2目的の問題
    problem = Problem(ny*(nx-1), 2, 1)
    # 最小化or最大化を設定
    problem.directions[:] = Problem.MAXIMIZE

    # 決定変数の範囲を設定
    int1 = Integer(0, 1)
    problem.types[:] = int1
    problem.constraints[:] = "<="+str(volume_frac)
    problem.function = objective
    problem.directions[:] = Problem.MAXIMIZE
    algorithm = NSGAII(problem, population_size=parent)
    algorithm.run(generation)

    # グラフを描画

    fig = plt.figure()
    plt.scatter([s.objectives[0] for s in algorithm.result],
                [s.objectives[1] for s in algorithm.result], c="blue", label="infeasible solution")

    plt.scatter([s.objectives[0] for s in algorithm.result if s.feasible],
                [s.objectives[1] for s in algorithm.result if s.feasible], c="red", label='feasible solution')

    # 非劣解をとりだす
    nondominated_solutions = nondominated(algorithm.result)
    plt.scatter([s.objectives[0] for s in nondominated_solutions if s.feasible],
                [s.objectives[1] for s in nondominated_solutions if s.feasible], c="green", label="pareto solution")
    plt.legend(loc='lower left')

    plt.xlabel("$E$")
    plt.ylabel("$G$")
    fig.savefig(os.path.join(PATH, "graph.png"))
    plt.close()

    for solution in [s for s in nondominated_solutions if s.feasible]:
        image_list = []
        for j in solution.variables:
            image_list.append(j)
        image = np.array(image_list).reshape(ny, nx-1)
        image = np.concatenate([image, np.ones((ny, 1))], 1)
        np.save(os.path.join(PATH, 'E_{}_G_{}.npy'.format(
            solution.objectives[0], solution.objectives[1])), image)

    convert_folder_npy_to_image(PATH)

    elapsed_time = time.time() - start

    with open("time.txt", mode='a') as f:
        f.writelines("grid_nx_{}_ny_{}_gen_{}_pa_{}:{}sec\n".format(
            nx, ny, generation, parent, elapsed_time))
