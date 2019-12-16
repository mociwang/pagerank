import numpy as np
from scipy.sparse import csc_matrix, save_npz, csr_matrix


def pageRank(G, s=.85, maxerr=.0001):
    """
    Computes the pagerank for each of the n states
    Parameters
    ----------
    G: matrix representing state transitions
    Gij is a binary value representing a transition from state i to j.
    s: probability of following a transition. 1-s probability of teleporting
    to another state.
    maxerr: if the sum of pageranks between iterations is bellow this we will
        have converged.
    """
    n = G.shape[0]
    # 将 G 压缩为 A
    A = csr_matrix(G, dtype=np.float)
    rsums = np.array(A.sum(1))[:, 0]
    ri, ci = A.nonzero()
    A.data /= rsums[ri]
    sink = rsums == 0
    # 计算PR值，直到满足收敛条件
    ro, r = np.zeros(n), np.ones(n)
    for _ in range(20):
        if np.sum(np.abs(r - ro)) < maxerr:
            break
        ro = r.copy()
        for i in range(0, n):
            Ai = np.array(A[:, i].todense())[:, 0]
            Di = sink / float(n)
            Ei = np.ones(n) / float(n)
            r[i] = ro.dot(Ai * s + Di * s + Ei * (1 - s))
        # print(_)
    # 归一化
    return r / float(sum(r))


if __name__ == '__main__':
    G = np.array([[0, 0, 1],
                  [1, 0, 0],
                  [1, 1, 0]])

    print(pageRank(G, s=0.85))
