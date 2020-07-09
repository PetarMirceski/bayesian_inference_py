import os
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm

from processing.bayesian_inference import BayesianInference
from processing.utils.constants import PHASE_OUTPUT_PATH

window_size = 57
overlap = 1
sampling_step = 1 / 100


def load_phases(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    infile = open(str(file_path), "rb")
    patient = pickle.load(infile, encoding="utf-8")
    phases = tuple(phase.reshape(1, -1) for _, phase in patient.items())
    return phases[0], phases[1]


def construct_name(file_path: Path) -> Path:
    file_path = file_path.with_suffix("")
    _, file_name = os.path.split(file_path)
    return Path(file_name)


def patient_bayes_main() -> None:
    FILE_PATHS = [path for path in PHASE_OUTPUT_PATH.rglob("*.pickle")]

    bayesian = BayesianInference(window_size, sampling_step, overlap)
    bayesian.check_folder()

    for file_path in tqdm(FILE_PATHS):
        file_name = construct_name(file_path)
        first_phase, second_phase = load_phases(file_path)
        bayesian(first_phase, second_phase)
        bayesian.save_inferred_data(file_name)

    print(bayesian.empty_report)


if __name__ == "__main__":
    patient_bayes_main()
