from typing import Tuple

import numpy as np

from processing.utils.array_manipulation import _norm


def dirc(c: np.ndarray) -> Tuple[float, float, float]:
    # TODO: TEST ME THIS IS A GOOD IMPLEMENTATION OF THE FUNCTION
    K = c.shape[0] // 2
    q1 = c[1:K]
    q2 = c[K + 1 :]

    assert q1.shape[0] == q2.shape[0]

    cpl_1 = _norm(q1)
    cpl_2 = _norm(q2)
    drc = (cpl_2 - cpl_1) / (cpl_1 + cpl_2)
    return cpl_1, cpl_2, drc
