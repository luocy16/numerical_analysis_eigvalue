import numpy as np
from QR import *


def test_matrix_generator(n):
    return \
        np.asarray([
            [2 if i == j else -1 if abs(i-j) == 1 else 0
         for i in range(0, n)]
        for j in range(0, n)
        ])

n = 30
ans = map(lambda x: format(x, '.7f'), -5 + basic_QR_method(test_matrix_generator(n) + 5 * np.eye(n)))
print(list(ans))
print(list(map(lambda x: format(x, '.7f'),np.sort(np.linalg.eigvals(test_matrix_generator(n))))))