import logging
import os
import pickle
import re
import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from processing.utils.constants import LOG_FILE_PATH, PROBE_COMBINATION_REGEX


def remove_and_create_folders(processed_signals: Path) -> None:
    UNPACK_FOLDERS = ["5-Hz", "6-Hz", "10-Hz"]
    if os.path.isdir(processed_signals):
        shutil.rmtree(processed_signals)
    for folder in UNPACK_FOLDERS:
        result_folder = processed_signals / folder
        result_folder.mkdir(parents=True, exist_ok=True)


def save_file(data: Dict[str, np.ndarray], location: Path) -> None:
    """SAVES TO TARGET LOCATION"""
    with open(str(location.with_suffix(".pickle")), "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def check_folder(path: Path) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def get_channel_combination(file: str) -> Tuple[int, int]:
    combination = re.findall(PROBE_COMBINATION_REGEX, file)[0]
    combination = combination.split(",")
    first = int(combination[0])
    second = int(combination[1])
    return first, second


def setup_logging() -> None:
    if os.path.exists(LOG_FILE_PATH):
        os.remove(LOG_FILE_PATH)
    logging.basicConfig(filename=LOG_FILE_PATH, level=logging.DEBUG)
