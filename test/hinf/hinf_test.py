import numpy as np
import cvxpy as cp
from scipy.optimize import minimize_scalar

A = np.block([
    [np.zeros((3, 3)), np.array([[1, 0, 0.17], [0, 1, 0], [0, 0, 1.02]])],
    [np.zeros((3, 3)), np.array([[-11.84, 0, 0], [0, -11.84, 0], [0, 0, -6.6]])]
])

B = np.vstack((
    np.zeros((3, 4)),
    np.array([
        [0, -52.04, 0, 52.04],
        [-52.04, 0, 52.04, 0],
        [4.31, -4.31, 4.31, -4.31]
    ])
))

ne = 3
At = np.block([
    [A, np.zeros((A.shape[0], ne))],
    [np.eye(ne), np.zeros((ne, A.shape[0] - ne)), np.zeros((ne, ne))]
])
Bt = np.vstack([B, np.zeros((ne, B.shape[1]))])

nx, nu = Bt.shape
nz = nx
N = 2

O1 = np.diag([1, 1, 1, 1])
O2 = np.diag([1, 1, 0.5, 1])
O_arr = [O1, O2]

l1_func = lambda t: np.exp(-t)
l2_func = lambda t: 1 - l1_func(t)

O_func = lambda t: l1_func(t) * O1 + l2_func(t) * O2

B_arr = [Bt @ O1, Bt @ O2]
H = Bt

E1 = np.diag((6, 6, 6, 0.01, 0.01, 0.01, 5, 5, 5))
E2 = np.diag((1, 1, 1, 0.01, 0.01, 0.01, 5, 5, 5))
E_arr = [E1, E2]

F1 = 2 * np.tile(
    np.array([
        [0, -1, 0, 1],
        [-1, 0, 1, 0],
        [1, 1, 1, 1]
    ]),
    reps=(3, 1)
)
F2 = 2 * np.tile(
    np.array([
        [0, -1, 0, 1],
        [-1, 0, 0.1, 0],
        [1, 1, 0.1, 1]
    ]),
    reps=(3, 1)
)
F_arr = [F1, F2]

G1 = 0.05 * F1
G2 = 0.05 * F1
G_arr = [G1, G2]


def He(X):
    """Hermitian operator"""
    return X + X.T


def symm(*arr, size=None):
    size = int(size or - 0.5 + np.sqrt(0.5**2 + 2 * len(arr)))
    assert len(arr) == size * (size + 1) / 2
    barr = np.zeros((size, size), dtype="O")
    indl = np.tril_indices(size)
    barr[indl] = arr
    barr.T[indl] = list(map(cp.transpose, barr[indl]))
    return cp.bmat(barr.tolist())


S_arr = [cp.Variable((nx, nx), PSD=True) for _ in range(N)]
Y_arr = [cp.Variable((nu, nx)) for _ in range(N)]
rho = cp.Variable(nonneg=True)
gamma_arr = [cp.Parameter(nonneg=True, value=400) for _ in range(N)]

LMI_const = []
for i in range(2):
    Si = S_arr[i]
    Yi = Y_arr[i]
    Ei = E_arr[i]
    Fi = F_arr[i]
    Gi = G_arr[i]
    gammai = gamma_arr[i]
    for j in range(2):
        Sj = S_arr[j]
        Bj = B_arr[j]
        LMI_const.append(symm(
            He(Si @ At.T + Yi.T @ Bj.T) - gammai * Si,
            H.T,
            - rho * np.eye(nu),
            Ei @ Si + Fi @ Yi,
            Gi,
            - np.eye(nz),
            gammai * Si,
            np.zeros((nx, nu)),
            np.zeros((nx, nz)),
            - gammai * Sj
        ) << 0)


obj = cp.Minimize(rho)
prob = cp.Problem(obj, LMI_const)
# prob.solve()


def func(val):
    for gamma in gamma_arr:
        gamma.value = val
    prob.solve()
    return rho.value


# res = minimize_scalar(func, method="golden", options={"disp": 1})
res = minimize_scalar(
    func, method="bounded", bounds=(30, 50), options={"verbose": True})
