import numpy as np

def Jacobi (A):
    outer_iter_time = 1000
    for i in range(0, outer_iter_time):
        p_max = 0
        q_max = 1
        max_elem = abs(A[0,1])
        for p in range(0, A.shape[0]):
            # 消掉元素j
           for q in range(p+1, A.shape[0]):
                if max_elem < abs(A[p, q]):
                   max_elem = abs(A[p, q])
                   p_max = p
                   q_max = q

        p = p_max
        q = q_max
        r = (A[p, p] - A[q, q]) / (2 * A[p, q])
        t = 1 / (abs(r) + np.sqrt(1 + r ** 2)) * (1 if r >= 0 else -1)
        c = 1 /np.sqrt(1 + t ** 2)
        s = c * t
        J = np.eye(A.shape[0])
        J[p, p] = J[q, q] = c
        J[p, q] = s
        J[q, p] = -s
        A = np.dot(np.dot(J, A), J.T)
    return np.sort([A[i, i] for i in range(0, A.shape[0])])

