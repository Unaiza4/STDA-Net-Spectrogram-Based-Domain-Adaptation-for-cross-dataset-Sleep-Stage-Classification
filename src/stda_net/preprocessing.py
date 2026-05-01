import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from math import gcd
from typing import List, Optional, Sequence, Tuple

import mne
import numpy as np
from scipy.signal import butter, filtfilt, resample_poly


@dataclass
class PreprocessConfig:
    target_fs: int = 100
    epoch_sec: int = 30
    wake_edge_minutes: int = 30

    # EEG channels
    sleepedf_channel: str = "EEG Fpz-Cz"
    shhs_channel_candidates: Sequence[str] = ("EEG", "EEG(sec)", "EEG 2", "EEG2")

    # Preprocessing: 4th-order Butterworth bandpass filter
    apply_bandpass: bool = True
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 30.0
    bandpass_order: int = 4


EDF_STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}

SHHS_STAGE_MAP = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 3,
    "5": 4,
}

STAGE_NAMES = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
}


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def normalize_signal_1d(x: np.ndarray) -> np.ndarray:
    """Min-max normalize one 30-second EEG epoch to [0, 1]."""
    x = np.asarray(x, dtype=np.float32)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    denom = x_max - x_min
    if denom < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - x_min) / denom).astype(np.float32)


def bandpass_filter(
    x: np.ndarray,
    fs: float,
    low_hz: float = 0.5,
    high_hz: float = 30.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter.
    """
    x = np.asarray(x, dtype=np.float64)
    nyquist = float(fs) / 2.0

    low = max(float(low_hz), 1e-6)
    high = min(float(high_hz), nyquist - 1e-6)

    if not (0.0 < low < high < nyquist):
        raise ValueError(
            f"Invalid bandpass range: low={low}, high={high}, nyquist={nyquist}, fs={fs}"
        )

    b, a = butter(order, [low / nyquist, high / nyquist], btype="bandpass")
    return filtfilt(b, a, x).astype(np.float32)


def resample_signal(x: np.ndarray, fs_orig: float, fs_target: float) -> np.ndarray:
    """Resample a 1D EEG signal using polyphase resampling."""
    x = np.asarray(x, dtype=np.float32)
    if abs(float(fs_orig) - float(fs_target)) < 1e-6:
        return x
    up = int(round(fs_target))
    down = int(round(fs_orig))
    factor = gcd(up, down)
    return resample_poly(x, up // factor, down // factor).astype(np.float32)


def pick_sleepedf_channel(raw: mne.io.BaseRaw, cfg: PreprocessConfig) -> str:
    if cfg.sleepedf_channel not in raw.ch_names:
        raise RuntimeError(f"Required channel {cfg.sleepedf_channel} not found. Available: {raw.ch_names}")
    return cfg.sleepedf_channel


def pick_shhs_channel(raw: mne.io.BaseRaw, cfg: PreprocessConfig) -> str:
    for channel in cfg.shhs_channel_candidates:
        if channel in raw.ch_names:
            return channel
    eeg_like = [ch for ch in raw.ch_names if "EEG" in ch.upper()]
    if eeg_like:
        return eeg_like[0]
    raise RuntimeError(f"No usable SHHS EEG channel found. Available: {raw.ch_names}")


def trim_wake_edges(
    epoch_info: List[Tuple[float, int]],
    wake_label: int = 0,
    edge_minutes: int = 30,
    epoch_sec: int = 30,
) -> List[Tuple[float, int]]:
    """Keep limited wake before sleep onset and after the last non-wake epoch."""
    if not epoch_info:
        return []

    edge_epochs = int((edge_minutes * 60) / epoch_sec)
    labels = [label for _, label in epoch_info]
    nonwake_idx = [i for i, label in enumerate(labels) if label != wake_label]

    if not nonwake_idx:
        return []

    start_idx = max(nonwake_idx[0] - edge_epochs, 0)
    end_idx = min(nonwake_idx[-1] + edge_epochs, len(epoch_info) - 1)
    return epoch_info[start_idx:end_idx + 1]


def parse_sleepedf_pairs(sleepedf_dir: str):
    files = os.listdir(sleepedf_dir)
    psg_files = sorted([f for f in files if f.endswith("-PSG.edf")], key=natural_key)
    pairs = []

    for psg in psg_files:
        prefix = psg.replace("-PSG.edf", "")
        subject_id = prefix[:6]
        night = prefix[6:]

        hyp_candidates = [
            f for f in files
            if f.startswith(subject_id) and "Hypnogram" in f and f.endswith(".edf")
        ]
        if not hyp_candidates:
            continue

        hyp = sorted(hyp_candidates, key=natural_key)[0]
        pairs.append((
            os.path.join(sleepedf_dir, psg),
            os.path.join(sleepedf_dir, hyp),
            subject_id,
            night,
        ))

    return pairs


def sleepedf_annotations_to_epochs(hypnogram_path: str, cfg: PreprocessConfig):
    annotations = mne.read_annotations(hypnogram_path)
    epoch_info = []

    for ann in annotations:
        desc = ann["description"]
        if desc not in EDF_STAGE_MAP:
            continue

        label = EDF_STAGE_MAP[desc]
        onset = float(ann["onset"])
        duration = float(ann["duration"])
        n_epochs = int(duration // cfg.epoch_sec)

        for i in range(n_epochs):
            epoch_info.append((onset + i * cfg.epoch_sec, label))

    return trim_wake_edges(epoch_info, 0, cfg.wake_edge_minutes, cfg.epoch_sec)


def find_shhs_xml(label_dir: str, subject_id: str, visit: str = "shhs1") -> Optional[str]:
    expected = f"{visit}-{subject_id}-profusion.xml"
    expected_path = os.path.join(label_dir, expected)
    if os.path.exists(expected_path):
        return expected_path

    candidates = [f for f in os.listdir(label_dir) if subject_id in f and f.endswith(".xml")]
    if candidates:
        return os.path.join(label_dir, sorted(candidates, key=natural_key)[0])

    return None


def parse_shhs_xml(xml_path: str, cfg: PreprocessConfig):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    epoch_info = []

    for event in root.iter("ScoredEvent"):
        event_type = event.findtext("EventType", "")
        concept = event.findtext("EventConcept", "")

        if "stage" not in event_type.lower() and "stage" not in concept.lower():
            continue

        stage = concept.split("|")[-1].strip() if "|" in concept else concept.strip()
        if stage not in SHHS_STAGE_MAP:
            continue

        label = SHHS_STAGE_MAP[stage]
        onset = float(event.findtext("Start", "0"))
        duration = float(event.findtext("Duration", "0"))
        n_epochs = int(duration // cfg.epoch_sec)

        for i in range(n_epochs):
            epoch_info.append((onset + i * cfg.epoch_sec, label))

    if not epoch_info:
        sleep_stages_node = root.find("SleepStages")
        if sleep_stages_node is not None:
            for idx, node in enumerate(sleep_stages_node.findall("SleepStage")):
                stage = (node.text or "").strip()
                if stage in SHHS_STAGE_MAP:
                    epoch_info.append((idx * cfg.epoch_sec, SHHS_STAGE_MAP[stage]))

    return trim_wake_edges(epoch_info, 0, cfg.wake_edge_minutes, cfg.epoch_sec)


def load_and_preprocess_recording(edf_path: str, dataset: str, cfg: PreprocessConfig):
    """
    Load EDF, select EEG channel, apply bandpass filtering,
    and resample to the target frequency.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

    if dataset.lower() == "sleepedf":
        channel = pick_sleepedf_channel(raw, cfg)
    else:
        channel = pick_shhs_channel(raw, cfg)

    signal = raw.get_data(picks=mne.pick_channels(raw.ch_names, include=[channel]))[0]
    fs_orig = float(raw.info["sfreq"])

    if cfg.apply_bandpass:
        signal = bandpass_filter(
            signal,
            fs=fs_orig,
            low_hz=cfg.bandpass_low_hz,
            high_hz=cfg.bandpass_high_hz,
            order=cfg.bandpass_order,
        )

    signal = resample_signal(signal, fs_orig=fs_orig, fs_target=cfg.target_fs)
    return signal, channel


def extract_epoch_signal(signal: np.ndarray, onset_sec: float, cfg: PreprocessConfig):
    """Extract and normalize one epoch from a preprocessed signal."""
    epoch_len = int(cfg.epoch_sec * cfg.target_fs)
    start = int(round(onset_sec * cfg.target_fs))
    end = start + epoch_len

    if start < 0 or end > len(signal):
        return None

    epoch = signal[start:end]
    if len(epoch) != epoch_len:
        return None

    return normalize_signal_1d(epoch)
