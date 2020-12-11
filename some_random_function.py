import numpy as np


def Himmelblau(m, temperature=100):
    return (
        (m[0, ...] ** 2 + m[1, ...] - 11) ** 2
        + (m[0, ...] + m[1, ...] ** 2 - 7) ** 2
    ) / temperature


def Himmelblau_grad(m, temperature=100):

    # Column vector
    gradient = np.zeros_like(m)

    # dX/dm0
    gradient[0, ...] = 2 * (
        2 * m[0, ...] * (m[0, ...] ** 2 + m[1, ...] - 11)
        + m[0, ...]
        + m[1, ...] ** 2
        - 7
    )

    # dX/dm1
    gradient[1, ...] = 2 * (
        m[0, ...] ** 2
        + 2 * m[1, ...] * (m[0, ...] + m[1, ...] ** 2 - 7)
        + m[1, ...]
        - 11
    )

    return gradient / temperature
