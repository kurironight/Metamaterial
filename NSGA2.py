from platypus import NSGAII, Problem, nondominated, Integer
import matplotlib.pyplot as plt
from FEM import calc_E, calc_G
import numpy as np
import os
from convert_npy_to_image import convert_folder_npy_to_image
import time

start = time.time()

nx = 20
ny = 20
volume_frac = 0.5
parent = 1000
generation = 50000
PATH = os.path.join("data", "nx_{}_ny_{}".format(nx, ny),
                    "gen_{}_pa_{}".format(generation, parent))
os.makedirs(PATH, exist_ok=True)
# 目的関数の設定


def objective(vars):
    rho = np.array(vars)
    rho = rho.reshape(ny, nx-1)
    rho = np.concatenate([rho, np.ones((ny, 1))], 1)
    volume = np.sum(rho)/(nx*ny)

    return [calc_E(rho), calc_G(rho)], [volume]


# 2変数2目的の問題
problem = Problem(ny*(nx-1), 2, 1)
# 最小化or最大化を設定
problem.directions[:] = Problem.MINIMIZE

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
# plt.show()
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
    f.writelines("nx_{}_ny_{}_gen_{}_pa_{}:{}sec\n".format(
        nx, ny, generation, parent, elapsed_time))
