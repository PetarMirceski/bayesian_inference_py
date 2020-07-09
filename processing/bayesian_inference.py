import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from scipy import linalg

from processing.utils.array_manipulation import (
    _get_quadrants,
    _has_matrix_inf,
    _has_matrix_nan,
    _midpoint,
    _parameter_sin_cos,
    _reconstruct_from_quadrants,
)
from processing.utils.constants import BAYESIAN_DATA_PATH
from processing.utils.data_utils import check_folder, setup_logging

epsilon: float = 0.01  # Only used for the unwrap of the phases
error_epsilon: float = 0.00001
max_loops: int = 500
L: int = 2


class BayesianInference:
    def __init__(
        self,
        window_size: int,
        sampling_step: float,
        overlap: float,
        propagation_constant: int = 1,
        fourier_base_order: int = 2,
    ) -> None:

        """Constructor for the bayesian inference.
            Make sure the signals are sampled consistently at
            a same sampling frequency.

            Parameters
            ----------
            window_size: int
                The size for splitting the time series in windows.
            sampling_step: float
                The frequency of sampling in seconds
            overlap: float
                The overlap of the signal windows. 1 means no overlap 0 means full
            propagation_constant: int
                The propagation constant of the bayesian inference
            fourier_base_order: int
                The order of the base function. Number of harmonics that are processed.
        """
        self.ph1: np.ndarray
        self.ph2: np.ndarray
        self.cc: List[np.ndarray]
        self.e: List[np.ndarray]

        # PARAMETERS OF THE BAYESIAN
        self.window_size: float = window_size / sampling_step
        self.stride: float = overlap * self.window_size
        self.sampling_step: float = sampling_step
        self.fourier_base_order: int = fourier_base_order

        self.pw_square: float = (
            self.window_size * self.sampling_step * propagation_constant
        ) ** 2

        self.M: int = int(2 * ((2 * self.fourier_base_order + 1) ** 2))
        self.K: int = int(self.M / L)
        setup_logging()

    def _unwrap_phases(self) -> None:
        """Check whether the phases of the signals are
           wrapped in the region [0, 2pi] and unwraps"""
        if (self.ph1 <= 2 * np.pi + epsilon).all() and (
            self.ph2 <= 2 * np.pi + epsilon
        ).all():
            self.ph1 = np.unwrap(self.ph1)
            self.ph2 = np.unwrap(self.ph2)

    def _set_dimensions(self) -> None:
        """Make sure the signals have the right shape"""
        phase1_shape = self.ph1.shape
        phase2_shape = self.ph2.shape
        if phase1_shape != phase2_shape:
            raise AssertionError("Phases of the signals missmatch")
        if phase1_shape[0] < phase1_shape[1]:
            self.ph1 = np.transpose(self.ph1)
            self.ph2 = np.transpose(self.ph2)

    def _derivative(
        self, first_series: np.ndarray, second_series: np.ndarray
    ) -> np.ndarray:
        """Calculates the derivative of the time series"""
        first_series_derivative = np.diff(first_series) / self.sampling_step
        second_series_derivative = np.diff(second_series) / self.sampling_step
        return np.vstack((first_series_derivative, second_series_derivative))

    def _construct_window(self, series: np.ndarray, nth_window: int) -> np.ndarray:
        """Extract a time window from the time series"""
        window = series[
            int(nth_window * self.stride) : int(
                nth_window * self.stride + self.window_size
            )
        ].T
        return window

    def _stop_condition(self, c_prior: np.ndarray, c_posterior: np.ndarray) -> float:
        """Calculate the stop condition for early return of the bayesian inference on one window"""
        return float(
            np.sum(
                ((c_prior - c_posterior) * (c_prior - c_posterior))
                / (c_posterior * c_posterior)
            )
        )

    def _r_help_variable(
        self,
        concentration_prior: np.ndarray,
        c_prior: np.ndarray,
        derivative: np.ndarray,
        model: np.ndarray,
        ed: np.ndarray,
    ) -> np.ndarray:
        return (
            concentration_prior[0] @ c_prior[:, 0:1]
            + concentration_prior[1] @ c_prior[:, 1:2]
            + self.sampling_step
            * (model @ ed.T - (1 / 2) * np.sum(derivative, axis=1).reshape(-1, 1))
        )

    # REVIEW: MAKE THIS MORE LITTERATE
    def _calculate_model(self, phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
        """Calculates the base function for a given time window"""
        model = np.zeros(shape=(self.K, phi1.shape[0]), dtype=np.float64)
        model[0, :] = 1
        br = 1

        for param_counter in range(2):
            for i in range(1, self.fourier_base_order + 1):
                if param_counter % 2 == 0:
                    sin, cos = _parameter_sin_cos(i * phi1)
                else:
                    sin, cos = _parameter_sin_cos(i * phi2)

                model[br, :] = sin
                model[br + 1, :] = cos
                br += 2

        for i in range(1, self.fourier_base_order + 1):
            for j in range(1, self.fourier_base_order + 1):
                weight_sum_phases = i * phi1 + j * phi2
                weight_diff_phases = i * phi1 - j * phi2

                model[br, :] = np.sin(weight_sum_phases)
                model[br + 1, :] = np.cos(weight_sum_phases)
                br = br + 2

                model[br, :] = np.sin(weight_diff_phases)
                model[br + 1, :] = np.cos(weight_diff_phases)
                br = br + 2
        return model

    # REVIEW: MAKE THIS MORE READABLE LITTERATE OR COMBINE IT IN A SANE WAY WITH THE SECOND DERIVATIVE
    def _first_variable_derivative_v(
        self, phi1: np.ndarray, phi2: np.ndarray
    ) -> np.ndarray:
        """Calculates the derivative of the first phase of the based model"""
        v = np.zeros(shape=(self.K, phi1.shape[0]), dtype=np.float64)
        br = 1
        for i in range(1, self.fourier_base_order + 1):
            v[br, :] = i * np.cos(i * phi1)
            v[br + 1, :] = -i * np.sin(i * phi1)
            br = br + 2

        # REVIEW: MAYBE USE SLICE
        for i in range(1, self.fourier_base_order + 1):
            v[br, :] = 0
            v[br + 1, :] = 0
            br = br + 2

        for i in range(1, self.fourier_base_order + 1):
            for j in range(1, self.fourier_base_order + 1):
                v[br, :] = i * np.cos(i * phi1 + j * phi2)
                v[br + 1, :] = -i * np.sin(i * phi1 + j * phi2)
                br = br + 2

                v[br, :] = i * np.cos(i * phi1 - j * phi2)
                v[br + 1, :] = -i * np.sin(i * phi1 - j * phi2)
                br = br + 2
        return v

    # REVIEW: MAKE THIS MORE READABLE LITTERATE OR COMBINE IT IN A SANE WAY WITH THE SECOND DERIVATIVE
    def _second_variable_derivative_v(
        self, phi1: np.ndarray, phi2: np.ndarray
    ) -> np.ndarray:
        """Calculates the derivative of the second phase of the based model"""
        v = np.zeros(shape=(self.K, phi2.shape[0]))
        br = 1

        for i in range(1, self.fourier_base_order + 1):
            v[br, :] = 0
            v[br + 1, :] = 0
            br = br + 2

        for i in range(1, self.fourier_base_order + 1):
            v[br, :] = i * np.cos(i * phi2)
            v[br + 1, :] = -i * np.sin(i * phi2)
            br = br + 2

        for i in range(1, self.fourier_base_order + 1):
            for j in range(1, self.fourier_base_order + 1):
                v[br, :] = j * np.cos(i * phi1 + j * phi2)
                v[br + 1, :] = -j * np.sin(i * phi1 + j * phi2)
                br = br + 2

                v[br, :] = -j * np.cos(i * phi1 - j * phi2)
                v[br + 1, :] = j * np.sin(i * phi1 - j * phi2)
                br = br + 2
        return v

    def _calculate_model_parameters(
        self,
        E: np.ndarray,
        model: np.ndarray,
        first_variable_derivative: np.ndarray,
        second_variable_derivative: np.ndarray,
        phi_derivative: np.ndarray,
        c_prior: np.ndarray,
        concentration_prior: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        concentration_posterior = np.zeros(shape=(4, self.M // 2, self.M // 2))
        K = self.K
        hppT = self.sampling_step * model @ model.T
        invr = linalg.pinv(E, check_finite=True)
        ed = invr @ phi_derivative

        concentration_prior = _get_quadrants(concentration_prior)
        for i in range(concentration_prior.shape[0]):
            concentration_posterior[i] = concentration_prior[i] + invr.ravel()[i] * hppT

        r1 = self._r_help_variable(
            concentration_prior[0:2], c_prior, first_variable_derivative, model, ed[0:1]
        )
        r2 = self._r_help_variable(
            concentration_prior[2:], c_prior, second_variable_derivative, model, ed[1:2]
        )

        concentration_posterior = _reconstruct_from_quadrants(concentration_posterior)
        c_posterior = (linalg.inv(concentration_posterior) @ np.vstack((r1, r2))).T
        c_posterior = np.vstack((c_posterior[:, :K], c_posterior[:, K:])).T
        return c_posterior, concentration_posterior

    def _calculate_noise_matrix(
        self, c_posterior: np.ndarray, phi_derivative: np.ndarray, model: np.ndarray
    ) -> np.ndarray:
        partial_error = phi_derivative - c_posterior @ model
        E = partial_error @ partial_error.T
        return (self.sampling_step / phi_derivative.shape[1]) * E

    def _propagation_function_XIpt(
        self, concentration_posterior: np.ndarray
    ) -> np.ndarray:
        """The gaussian of the posterior is convoluted with another
          gaussian which express the diffusion of the parameter"""
        inverse_concentration_posterior = linalg.inv(concentration_posterior)
        inv_diffusion = np.diagflat(
            np.diag(inverse_concentration_posterior) * self.pw_square
        )
        return linalg.inv(inverse_concentration_posterior + inv_diffusion)

    def _main_calculation(self) -> None:
        c_prior = np.zeros(shape=(int(self.M / L), L))
        concentration_prior = np.zeros(shape=(self.M, self.M))

        num_itterations: int = int(
            np.floor((self.ph1.shape[0] - self.window_size) / self.stride + 1)
        )

        for i in range(num_itterations):
            phase1_window = self._construct_window(self.ph1, i)
            phase2_window = self._construct_window(self.ph2, i)

            (c_prior, concentration_posterior, E, inf_error) = self._bayes_window(
                phase1_window, phase2_window, c_prior, concentration_prior
            )

            if inf_error:
                continue

            concentration_prior = self._propagation_function_XIpt(
                concentration_posterior
            )
            self.e.append(E)
            self.cc.append(np.ravel(c_prior.T))

    def _bayes_window(
        self,
        phase1_window: np.ndarray,
        phase2_window: np.ndarray,
        c_prior: np.ndarray,
        concentration_prior: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        # MIDPOINTS
        phi1_midpoint = _midpoint(phase1_window)
        phi2_midpoint = _midpoint(phase2_window)

        # TIME_DERIVATIVE
        phi_derivative = self._derivative(phase1_window, phase2_window)

        # FILL TEMPORARY VARIABLES TO BE USED BELLOW IN THE MAIN CALCULATIONS
        model = self._calculate_model(phi1_midpoint, phi2_midpoint)
        first_variable_derivative = self._first_variable_derivative_v(
            phi1_midpoint, phi2_midpoint
        )
        second_variable_derivative = self._second_variable_derivative_v(
            phi1_midpoint, phi2_midpoint
        )

        c_posterior = c_prior.copy()

        for loop in range(max_loops):
            E = self._calculate_noise_matrix(c_posterior.T, phi_derivative, model)

            if _has_matrix_inf(E) or _has_matrix_nan(E):
                # RETURN THE PARAMETERS WITH THE PRIOR CONCENTRATION IF ERROR IS PRESENT
                return (c_posterior, concentration_prior, E, True)

            (c_posterior, concentration_posterior) = self._calculate_model_parameters(
                E,
                model,
                first_variable_derivative,
                second_variable_derivative,
                phi_derivative,
                c_prior,
                concentration_prior,
            )

            if self._stop_condition(c_prior, c_posterior) < error_epsilon:
                return c_posterior, concentration_posterior, E, False
            c_prior = c_posterior.copy()

        return c_posterior, concentration_posterior, E, False

    def get_time_vector(self) -> np.ndarray:
        """Returns the time vector for plotting of the data inferred,
           needs to fit phases for inferring the time vector data"""
        return (
            np.arange(
                self.window_size / 2,
                (self.ph1.shape[0] - self.window_size / 2),
                self.stride,
            )
            * self.sampling_step
        )

    def check_folder(self) -> None:
        """Removes the data folder and creates a new one for saving"""
        check_folder(BAYESIAN_DATA_PATH)

    def save_inferred_data(self, file_name: Union[Path, str]) -> None:
        """Save the inferred data and handle failed results"""
        location = (BAYESIAN_DATA_PATH / file_name).with_suffix("")
        location = location.with_suffix(".pickle")

        if self.cc == []:
            logging.debug(f"failed: {str(file_name)}")
        else:
            data = {"cc": self.cc, "e": self.e, "tm": self.get_time_vector()}
            with open(str(location), "wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __call__(
        self, ph1: np.ndarray, ph2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # RESET THE PARAMETERS
        self.ph1 = ph1
        self.ph2 = ph2
        self.cc = []
        self.e = []
        self._unwrap_phases()
        self._set_dimensions()
        self._main_calculation()
        return self.cc, self.e
