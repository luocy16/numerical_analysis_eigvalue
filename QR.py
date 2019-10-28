import numpy as np


class QR:
    def __init__(self, matrix):
        self.matrix = matrix
        self.row, self.column = matrix.shape
        self.Q = np.eye(self.row)
        self.R = np.zeros((self.row, self.column))
        assert self.row == self.column

    def qr_decomposition(self):
        get_norm = lambda x:  np.sqrt(np.sum([i**2 for i in x]))
        for i in range(self.row - 1):
            self.R[:i, i] = self.matrix[:i, i]
            self.R[i, i] = get_norm(self.matrix[i:, i])
            diff = self.R[i:, i:i+1] - self.matrix[i:, i:i+1]
            norm_diff = get_norm(diff)
            v = diff/norm_diff
            H = np.eye(self.row)
            H[i:, i:] -= 2 * np.dot(v, v.T)
            self.matrix = np.dot(H, self.matrix)
            self.Q = np.dot(H, self.Q)
        self.R[:, -1] = self.matrix[:, -1]

    def get_qr(self):
        self.qr_decomposition()
        return self.Q.T, self.R


def basic_QR_method(A):
    history_list = []
    for i in range(600):
        Q, R = QR(A).get_qr()
        A = np.dot(R, Q)
        history_list.append([A[i, i] for i in range(A.shape[0])])

        if np.isnan(sum(history_list[-1])):
            print('迭代', i, '步后由于数值精度不够而终止')
            break
        if i % 10 == 0:
            print('finish percent {} %'.format(i / 600 * 100))
    return np.sort(history_list[-2])