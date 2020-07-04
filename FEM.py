import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

nx = 2
ny = 2

rho = np.ones((nx, ny))
po = 0.3
E = 1


def run():
    K = np.zeros((2 * (nx + 1) * (ny + 1), 2 * (nx + 1)
                  * (ny + 1)), dtype=np.float64)
    U = np.zeros((2 * (nx + 1) * (ny + 1)), dtype=np.float64)
    for y in range(1, ny+1):
        for x in range(1, nx+1):
            n1 = (ny + 1) * (x - 1) + y
            n2 = (ny + 1) * x + y
            # 要素を構成する要素番号(x,yそれぞれを含む，奇数がx，偶数がy)
            elem = np.array([2*n1-1, 2*n1, 2*n2-1, 2*n2,
                             2*n2+1, 2*n2+2, 2*n1+1, 2*n1+2])
            xc = [x - 1, x, x, x - 1]  # x座標の範囲は0~nx
            yc = [y - 1, y - 1, y, y]  # y座標の範囲は0~ny
            K_mat = Kmat_pl4(xc, yc, po)
            elem -= 1  # indexを指定する為
            K[np.ix_(elem, elem)] += K_mat  # 深いコピーになる

    # Boundary Condition
    F = np.zeros(2 * (nx + 1) * (ny + 1), dtype=np.float64)
    F[2*(ny+1)*(nx)+2] = 1
    FixDOF = [2 * ny + 1, 2 * (ny + 1), 2 * (nx + 1) * (ny + 1)]

    # 要素番号は1~2 * (nx + 1) * (ny + 1)まで
    FreeDOF = list(range(1, 2 * (nx + 1) * (ny + 1)+1))
    for i in FixDOF:
        FreeDOF.remove(i)
    # indexを示す為
    FreeDOF = np.array(FreeDOF) - 1
    FixDOF = np.array(FixDOF) - 1
    target_K = csr_matrix(K[np.ix_(FreeDOF, FreeDOF)])
    U[FreeDOF] = spsolve(
        target_K,   F[FreeDOF], use_umfpack=True)
    U[FixDOF] = 0


def dmat_pl4(E, po, nstr=1):
    # Dマトリックス作成
    # nstr=0の時，平面歪み，1の時，平面応力
    D_mat = np.zeros((3, 3), dtype=np.float64)
    if nstr == 0:  # plane strain
        D_mat[0, 0] = 1-po
        D_mat[0, 1] = po
        D_mat[1, 0] = po
        D_mat[1, 1] = 1-po
        D_mat[2, 2] = 0.5*(1-2*po)
        D_mat = E/(1+po)/(1-2*po)*D_mat
    else:  # plane stress
        D_mat[0, 0] = 1
        D_mat[0, 1] = po
        D_mat[1, 0] = po
        D_mat[1, 1] = 1
        D_mat[2, 2] = 0.5*(1-po)
        D_mat = E/(1-po**2)*D_mat
    return D_mat


def Kmat_pl4(xc, yc, po, t=1):
    """各四角形事のKマトリックスを作成

    Args:
        xc ([list]): 要素を構成するx座標のリスト
        yc ([list]): 要素を構成するy座標のリスト
        po ([type]): ポアソン比
        t (int, optional): z方向の要素の厚み. Defaults to 1.
    """
    K_mat = np.zeros((8, 8))
    D_mat = dmat_pl4(E, po)
    point, weight = gauss_point()
    for i in range(4):
        B_mat, detJ = bmat_pl4(point[i, 0], point[i, 1], xc, yc)
        K_mat = K_mat + weight[i] * \
            np.dot(B_mat.T, np.dot(D_mat, B_mat)) * t * detJ
    return K_mat


def gauss_point():
    point = np.array([[-3 ** (1 / 2), -3 ** (1 / 2)],
                      [3 ** (1 / 2), -3 ** (1 / 2)],
                      [3 ** (1 / 2), 3 ** (1 / 2)],
                      [-3 ** (1 / 2), 3 ** (1 / 2)]], dtype=np.float64)
    weight = [1, 1, 1, 1]
    return point, weight


def bmat_pl4(a, b, xc, yc):
    # Bマトリックスとヤコビアンを導出
    # aとbには,ガウスpointが入る
    x1, x2, x3, x4 = xc
    y1, y2, y3, y4 = yc
    bm = np.zeros((3, 8))
    # dN/da,dN/db
    dn1a = -0.25*(1.0-b)
    dn2a = 0.25*(1.0-b)
    dn3a = 0.25*(1.0+b)
    dn4a = -0.25*(1.0+b)
    dn1b = -0.25*(1.0-a)
    dn2b = -0.25*(1.0+a)
    dn3b = 0.25*(1.0+a)
    dn4b = 0.25*(1.0-a)
    # Jacobi matrix and det(J)
    J11 = dn1a*x1+dn2a*x2+dn3a*x3+dn4a*x4
    J12 = dn1a*y1+dn2a*y2+dn3a*y3+dn4a*y4
    J21 = dn1b*x1+dn2b*x2+dn3b*x3+dn4b*x4
    J22 = dn1b*y1+dn2b*y2+dn3b*y3+dn4b*y4
    detJ = J11*J22-J12*J21
    # [B]=[dN/dx][dN/dy]
    bm[0, 0] = J22*dn1a-J12*dn1b
    bm[0, 2] = J22*dn2a-J12*dn2b
    bm[0, 4] = J22*dn3a-J12*dn3b
    bm[0, 6] = J22*dn4a-J12*dn4b
    bm[1, 1] = -J21*dn1a+J11*dn1b
    bm[1, 3] = -J21*dn2a+J11*dn2b
    bm[1, 5] = -J21*dn3a+J11*dn3b
    bm[1, 7] = -J21*dn4a+J11*dn4b
    bm[2, 0] = -J21*dn1a+J11*dn1b
    bm[2, 1] = J22*dn1a-J12*dn1b
    bm[2, 2] = -J21*dn2a+J11*dn2b
    bm[2, 3] = J22*dn2a-J12*dn2b
    bm[2, 4] = -J21*dn3a+J11*dn3b
    bm[2, 5] = J22*dn3a-J12*dn3b
    bm[2, 6] = -J21*dn4a+J11*dn4b
    bm[2, 7] = J22*dn4a-J12*dn4b
    bm = bm/detJ
    return bm, detJ


run()
