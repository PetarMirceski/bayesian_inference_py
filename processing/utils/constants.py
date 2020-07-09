from itertools import combinations
from pathlib import Path

"""THIS FOLDER CONTAINS ALL THE PROJECT CONSTANTS"""

# CONSTANTS
SAMPLING_FREQ = 2048  # THE ORIGINAL SAMPLING FREQUENCY
RE_SAMPLE_FREQ = 100  # THE RESAMPLING FREQUENCY

# THE ORDER OF THE CHANNELS IN THE EDF FILES BY INDEX
CHANNELS = [
    "EEG PO8",
    "EEG PO7",
    "EEG P8",
    "EEG P7",
    "EEG O1",
    "EEG O2",
    "EEG PO3",
    "EEG PO4",
]
LEFT_HALF_PROBES = ["O1", "PO7", "PO3", "P7"]
RIGHT_HALF_PROBES = ["O2", "PO8", "PO4", "P8"]

# PROBE COMBINATIONS
_COMBINATIONS = range(1, 9)
ALL_COMBINATIONS = [(comb[0], comb[1]) for comb in combinations(_COMBINATIONS, 2)]

# UNPACKING INPUTS
EDF_FILE_PATH = Path("./input_data/orig_data/")
PATIENT_CSV_PATH = Path("./input_data/RECORDS")

# UNPACKING OUTPUTS
UNPACKED_SIGNALS_PATH = Path("./output_data/unpacked_signals")

# LEFT RIGHT PROBE TEST
PROBE_PROCESSED_SIGNALS = Path("./output_data/left_right_unpacked_signals")

# PATH TO THE FIRST SUBJECT IN THE 5-HZ file
KURAMOTO_INDEX_SUBJECT_PATH = [
    Path("./output_data/phases/5-Hz/rsvp_5Hz_02a(1,4).pickle"),
    Path("./output_data/phases/5-Hz/rsvp_5Hz_02a(7,8).pickle"),
]

# FILE PATH FOR IMAGE SAVING OF THE UNIT CIRCLE
PLOT_KURAMOTO_PATH = Path("./output_data/sync_index")

# PHASE OUTPUT FOLDER
PHASE_OUTPUT_PATH = Path("./output_data/phases")

# BAYESIAN PLOTS OUTPUT FOLDERS
BAYESIAN_PLOTS_PATH = Path("./output_data/bayesian/plots")

# BAYESIAN DATA OUTPUT FOLDER
BAYESIAN_DATA_PATH = Path("./output_data/bayesian/data")

# REGEX THAT MATCHES PROBE COMBINATION (BETWEEN BRACKETS)
PROBE_COMBINATION_REGEX = r"\((.*?)\)"

# FREQ COUPLING MAPS
FREQUENCY_MAP_PATH = Path("./output_data/plotting/freq_coupling_maps")

# STATE COUPLING MAPS
STATE_MAP_PATH = Path("./output_data/plotting/state_coupling_maps")

# LOGGING PATH
LOG_FILE_PATH = Path("./output_data/failed_experiment_logging.log")


# HILBERT PLOTTING PATH
HILBERT_SUBJECT_PATH = Path("./output_data/unpacked_signals/5-Hz/rsvp_5Hz_02a.pickle")
HILBERT_FIG_PATH = Path("./output_data/plotting/hilbert_plots/")
