from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from numba import njit


# TODO: REMOVE THE UNWANTED COORDINATES
def co_hilbproto(
    signal: np.ndarray,
    fignum: Optional[int] = None,
    x0: float = 0,
    y0: float = 0,
    ntail: int = 1000,
) -> np.ndarray:
    """ INPUT:  x      is scalar timeseries,
                x0,y0  are coordinates of the origin (by default x0=0, y0=0)
                ntail  is the number of points at the ends to be cut off,
                       by default ntail=1000
        Output: theta is the protophase in 0,2pi interval
                minamp is the minimal instantaneous amplitude over the average
                instantaneous amplitude
    """
    hilbert_tf = scipy.signal.hilbert(signal)
    hilbert_tf = hilbert_tf[ntail : hilbert_tf.shape[0] - ntail]
    hilbert_tf.astype(np.complex64)
    hilbert_tf = hilbert_tf - np.mean(hilbert_tf)
    if fignum is not None and fignum > 0:
        plt.figure(fignum)
        plt.plot(np.real(hilbert_tf), np.imag(hilbert_tf))
        plt.plot(x0, y0, "ro")
        plt.xlabel("signal")
        plt.ylabel("HT(signal)")
        plt.title("Hilbert embedding")
        plt.show()
    hilbert_tf = hilbert_tf - x0 - 1j * y0
    hilbert_tf.astype(np.complex64)
    theta = np.angle(hilbert_tf).astype(np.float32)
    theta = theta % (2 * np.pi)
    return theta


@njit
def co_fbtransf1(
    theta: np.ndarray, nfft: int = 80, alpha: float = 0.05, ngrid: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Fourier series based transformation
        protophase theta --> phase phi for one oscillator.
        Input parameters:
            theta:  protophase
            nfft:   max number of Fourier harmonics,
                    default value is 80
            alpha: smoothing coefficient, by default alpha=0.05
        Output:
                [phi,arg,sigma] = co_fbtransf1(...) if also the transformation
                function sigma is required; it can be plotted as
                plot(arg,sigma); sigma is computed on the grid.
                Default grid size is 50.
    """
    Spl = np.zeros(shape=(nfft, 1), dtype=np.complex64)
    al2 = alpha ** 2
    npt = theta.shape[0]
    for i in range(1, nfft + 1):
        Spl[i - 1] = np.sum(np.exp(-1j * i * theta)) / npt
    phi = np.copy(theta).astype(np.float64)
    arg = np.arange(0, ngrid).reshape(1, -1)
    arg = arg * np.pi * 2 / (ngrid - 1)
    arg = arg.T
    sigma = np.ones(shape=(ngrid, 1))
    for i in range(1, nfft + 1):
        kernel = np.exp(-0.5 * i ** 2 * al2)
        sigma = sigma + kernel * 2 * np.real(Spl[i - 1] * np.exp(1j * i * arg))
        phi = phi + kernel * 2 * np.imag(Spl[i - 1] * (np.exp(1j * i * theta) - 1) / i)

    return phi, arg, sigma


def series_to_phase(
    first_signal: np.ndarray, second_signal: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    first_signal_hilbert = co_hilbproto(first_signal, -1, 0, 0, 0)
    second_signal_hilbert = co_hilbproto(second_signal, -1, 0, 0, 0)
    first_signal_phase, _, _ = co_fbtransf1(first_signal_hilbert)
    second_signal_phase, _, _ = co_fbtransf1(second_signal_hilbert)
    first_signal_phase = np.unwrap(first_signal_phase)
    second_signal_phase = np.unwrap(second_signal_phase)
    return (first_signal_phase.reshape(1, -1), second_signal_phase.reshape(1, -1))
