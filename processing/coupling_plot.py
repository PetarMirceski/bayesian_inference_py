from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from processing.utils.constants import BAYESIAN_PLOTS_PATH


def construct_coupling_function(
    cc: List[np.ndarray], fourier_base_order: int, index: int = -1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(0, 2 * np.pi, 0.13)
    q1 = np.zeros(shape=(t.shape[0], t.shape[0]))
    q2 = q1.copy()

    # NOTE: INDEX IS WHICH TIME WINDOW TO CONSTRUCT
    #       COUPLING IF -1 CONSTRUCT THE MEAN COUPLING
    if index == -1:
        cc = np.array(cc).mean(0)
    else:
        cc = cc[index]

    K = len(cc) // 2

    for i in range(len(t)):
        for j in range(len(t)):
            br = 1
            for _ in range(2):
                for idx in range(1, fourier_base_order + 1):
                    q1[i, j] = (
                        q1[i, j]
                        + cc[br] * np.sin(idx * t[i])
                        + cc[br + 1] * np.cos(idx * t[i])
                    )
                    q2[i, j] = (
                        q2[i, j]
                        + cc[K + br] * np.sin(idx * t[j])
                        + cc[K + br + 1] * np.cos(idx * t[j])
                    )
                    br += 2

            for ii in range(1, fourier_base_order + 1):
                for jj in range(1, fourier_base_order + 1):
                    phase_sum = ii * t[i] + jj * t[j]
                    phase_difference = ii * t[i] - jj * t[j]

                    q1[i, j] = (
                        q1[i, j]
                        + cc[br] * np.sin(phase_sum)
                        + cc[br + 1] * np.cos(phase_sum)
                    )
                    q2[i, j] = (
                        q2[i, j]
                        + cc[K + br] * np.sin(phase_sum)
                        + cc[K + br + 1] * np.cos(phase_sum)
                    )
                    br += 2

                    q1[i, j] = (
                        q1[i, j]
                        + cc[br] * np.sin(phase_difference)
                        + cc[br + 1] * np.cos(phase_difference)
                    )
                    q2[i, j] = (
                        q2[i, j]
                        + cc[K + br] * np.sin(-phase_difference)
                        + cc[K + br + 1] * np.cos(-phase_difference)
                    )
                    br += 2

    return q1.T, q2, t


def _transparent_plot_map(ax: matplotlib.axes) -> None:
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.view_init(elev=50, azim=40)


def plot_coupling_function(
    cc: Union[np.ndarray, List[np.ndarray]],
    fourier_base_order: int = 2,
    index: int = -1,
    file_name: Optional[Union[str, Path]] = None,
) -> None:
    q1, q2, t = construct_coupling_function(cc, fourier_base_order)
    T1, T2 = np.meshgrid(t.copy(), t.copy())

    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(T1, T2, q1, cmap=matplotlib.cm.hot)
    _transparent_plot_map(ax)

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.plot_surface(T1, T2, q2, cmap=matplotlib.cm.hot)
    _transparent_plot_map(ax)

    if file_name is not None:
        location = BAYESIAN_PLOTS_PATH / file_name
        if location.suffix == "":
            location = location.with_suffix(".png")
        plt.savefig(str(location))
    else:
        plt.show()

    plt.close()
