# TODO: MAKE THIS WORK WITH PICKLE MATLAB ISN'T WELCOME HERE
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

from processing.net_coupling import dirc
from processing.utils.constants import ALL_COMBINATIONS
from processing.utils.data_utils import get_channel_combination


def load_data(file_path: Path) -> np.ndarray:
    infile = open(str(file_path), "rb")
    patient = pickle.load(infile, encoding="utf-8")
    return np.array(patient["cc"])


def construct_mean_maps(
    frequency_mean_dict: Dict[Tuple[int, int], np.ndarray], fourier_base_order: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    color_map_2_1 = np.zeros((8, 8))  # strength 2 to 1
    color_map_1_2 = np.zeros((8, 8))  # strength 1 to 2
    direction_map = np.zeros((8, 8))  # direction
    for first_channel, second_channel in frequency_mean_dict.keys():
        mean_cc = frequency_mean_dict[(first_channel, second_channel)]

        cpl_1, cpl_2, drc = dirc(mean_cc)

        color_map_2_1[first_channel - 1, second_channel - 1] = cpl_1

        color_map_1_2[first_channel - 1, second_channel - 1] = cpl_2

        direction_map[first_channel - 1, second_channel - 1] = drc
        direction_map[second_channel - 1, first_channel - 1] = drc
    return color_map_2_1, color_map_1_2, direction_map


def mean_parameter_calculation(
    file_paths: Path, fourier_base_order: int = 2
) -> Dict[Tuple[int, int], np.ndarray]:

    experiment_combination = {
        combination: np.zeros((50)) for combination in ALL_COMBINATIONS
    }

    experiment_combination_counter = {
        combination: 0 for combination in ALL_COMBINATIONS
    }

    for file_path in tqdm(file_paths):
        combination = get_channel_combination(str(file_path))
        cc = load_data(file_path)

        # NOTE: THIS IS A TEST FOR ABNORMAL DATA
        #      SOME EEG FILES HAD INSANE COUPLING DATA
        cc = cc.mean(0)
        cpl_1, cpl_2, drc = dirc(cc)

        if cpl_1 > 10000 or cpl_2 > 10000:
            continue

        experiment_combination[combination] += cc
        experiment_combination_counter[combination] += 1

    for combination in experiment_combination_counter.keys():
        experiment_combination[combination] = (
            experiment_combination[combination]
            / experiment_combination_counter[combination]
        )

    return experiment_combination
