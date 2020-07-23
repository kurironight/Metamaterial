import numpy as np


def make_bar_structure(x_size, y_size, edges):
    """edgesを基に，バーを用いた構造を作成する

    Args:
        x_size (int): 目標とする1/4スケールの構造物のx方向のブロック数
        y_size (int): 目標とする1/4スケールの構造物のy方向のブロック数
        edges (list): [エッジの始点，終点，太さ]のリスト

    Returns:
        np.array (y_size * x_size): バー構造
    """
    rho = np.zeros([y_size, x_size], dtype=np.float64)
    for edge in edges:
        put_bar(rho, edge[0], edge[1], edge[2])
    rho = rho.astype(np.int32)
    return rho


def make_6_bar_edges(X, Y, y_1, y_2, y_3, x_4, nodes, widths):
    """６個のバーそれぞれに４個のノードを配置したもののedgeを作成

    Args:
        X (int): 目標とする1/4スケールの構造物のx方向のブロック数
        Y (int): 目標とする1/4スケールの構造物のy方向のブロック数
        y_1 (int): 左のノードのy座標
        y_2 (int): 右のノード1のy座標
        y_3 (int): 右のノード2のy座標
        x_4 (int): 上のノードのx座標
        nodes (list): (6*3,2) それぞれのノードの座標を示している．x,yの順．
        widths (list): (6*4) それぞれのエッジの太さを示す．
    """
    # edge1
    edges = [[[1, y_1], nodes[0], widths[0]]]
    for i in range(0, 2):
        edges.append([nodes[i], nodes[i + 1], widths[i + 1]])
    edges.append([nodes[2], [X, y_2], widths[3]])

    # edge2
    edges.append([[X, y_2], nodes[3], widths[4]])
    for i in range(3, 5):
        edges.append([nodes[i], nodes[i + 1], widths[i + 2]])
    edges.append([nodes[5], [X, y_3], widths[7]])

    # edge3
    edges.append([[X, y_3], nodes[6], widths[8]])
    for i in range(6, 8):
        edges.append([nodes[i], nodes[i + 1], widths[i + 3]])
    edges.append([nodes[8], [x_4, 1], widths[11]])

    # edge4
    edges.append([[x_4, 1], nodes[9], widths[12]])
    for i in range(9, 9+2):
        edges.append([nodes[i], nodes[i + 1], widths[i + 4]])
    edges.append([nodes[9 + 2], [1, y_1], widths[15]])

    # edge5
    edges.append([[1, y_1], nodes[12], widths[16]])
    for i in range(12, 12+2):
        edges.append([nodes[i], nodes[i + 1], widths[i + 5]])
    edges.append([nodes[12 + 2], [X, y_3], widths[19]])

    # edge6
    edges.append([[x_4, 1], nodes[15], widths[20]])
    for i in range(15, 15+2):
        edges.append([nodes[i], nodes[i + 1], widths[i + 6]])
    edges.append([nodes[15 + 2], [X, y_2], widths[23]])

    return edges


def put_bar(rho, start_point, end_point, width):
    assert start_point[0] >= 1 and start_point[0] <= rho.shape[1], 'start_point x index {} must be 1~{}'.format(
        start_point[0], rho.shape[1])
    assert start_point[1] >= 1 and start_point[1] <= rho.shape[0], 'start_point y index {} must be 1~{}'.format(
        start_point[1], rho.shape[0])
    assert end_point[0] >= 1 and end_point[0] <= rho.shape[1], 'end_point x index {} must be 1~{}'.format(
        end_point[0], rho.shape[1])
    assert end_point[1] >= 1 and end_point[1] <= rho.shape[0], 'end_point y index {} must be 1~{}'.format(
        end_point[1], rho.shape[0])
    start_point = np.array(start_point)
    end_point = np.array(end_point)
    # 端点が，始点終点が一緒の場合は何もしない
    if np.all(start_point == end_point):
        return rho
    else:
        edge_point1 = end_point+(end_point-start_point) / \
            np.linalg.norm(end_point - start_point, ord=2)*0.5
        edge_point2 = start_point + \
            (start_point-end_point) / \
            np.linalg.norm(end_point - start_point, ord=2)*0.5
    x_index = np.arange(1, rho.shape[1]+1, dtype=np.float64)
    y_index = np.arange(1, rho.shape[0]+1, dtype=np.float64)
    xx, yy = np.meshgrid(x_index, y_index)
    if (end_point[0]-start_point[0]) != 0:  # x=8等のような直線の式にならない場合
        m = (end_point[1]-start_point[1])/(end_point[0]-start_point[0])
        n = end_point[1]-m*end_point[0]
        d = np.abs(yy-m*xx-n)/np.sqrt(1+np.power(m, 2))
        # 垂線の足を求める
        X = (m*(yy-n)+xx)/(np.power(m, 2)+1)
        Y = m*X+n
        # バーを配置できる条件を満たすインデックスを求める
        X_on_segment = np.logical_and(min(
            edge_point1[0], edge_point2[0]) <= X, X <= max(edge_point1[0], edge_point2[0]))
        Y_on_segment = np.logical_and(min(
            edge_point1[1], edge_point2[1]) <= Y, Y <= max(edge_point1[1], edge_point2[1]))
        on_segment = np.logical_and(X_on_segment, Y_on_segment)
        in_distance = d <= width/2
        meet_index = np.logical_and(on_segment, in_distance)
    else:  # x=8等の場合
        d = np.abs(end_point[0]-xx)
        # 垂線の足を求める
        X = end_point[0]
        Y = yy
        # バーを配置できる条件を満たすインデックスを求める
        Y_on_segment = np.logical_and(min(
            edge_point1[1], edge_point2[1]) <= Y, Y <= max(edge_point1[1], edge_point2[1]))
        in_distance = d <= width/2
        meet_index = np.logical_and(Y_on_segment, in_distance)
    rho[meet_index] = 1
    return rho
