import glob
import os
import re
from dataclasses import dataclass

import numpy as np
from scipy.signal import get_window, stft

from .preprocessing import (
    PreprocessConfig,
    STAGE_NAMES,
    extract_epoch_signal,
    find_shhs_xml,
    load_and_preprocess_recording,
    parse_shhs_xml,
    parse_sleepedf_pairs,
    sleepedf_annotations_to_epochs,
)


@dataclass
class SpectrogramConfig:
    target_fs: int = 100
    n_fft: int = 256
    win_sec: float = 2.0
    overlap: float = 0.5
    db_min: float = -80.0
    db_max: float = 0.0


class SpectrogramTransformer:
    def __init__(self, cfg: SpectrogramConfig):
        self.cfg = cfg
        self.win_length = int(round(cfg.win_sec * cfg.target_fs))
        self.hop_length = int(round(self.win_length * (1.0 - cfg.overlap)))
        if self.hop_length <= 0:
            raise ValueError("hop_length must be greater than zero.")
        self.window = get_window("hamming", self.win_length, fftbins=True)

    def epoch_to_spectrogram(self, epoch_signal: np.ndarray) -> np.ndarray:
        _, _, zxx = stft(
            epoch_signal,
            fs=self.cfg.target_fs,
            window=self.window,
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.cfg.n_fft,
            boundary=None,
            padded=False,
        )
        magnitude = np.abs(zxx)
        db_spec = 20.0 * np.log10(np.maximum(magnitude, 1e-10))
        db_spec = np.clip(db_spec, self.cfg.db_min, self.cfg.db_max)
        spec = (db_spec - self.cfg.db_min) / (self.cfg.db_max - self.cfg.db_min)
        spec = np.nan_to_num(spec, nan=0.0, posinf=1.0, neginf=0.0)
        return spec.astype(np.float32)


def save_epoch_spectrogram(spec, output_root, dataset_name, subject_name, stage_name, epoch_index):
    out_dir = os.path.join(output_root, dataset_name, subject_name, stage_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"epoch_{epoch_index:04d}.npy")
    np.save(out_path, spec.astype(np.float32))


def save_subject_metadata(output_root, dataset_name, subject_name, channel_name, saved_total, pre_cfg, spec_cfg):
    subject_dir = os.path.join(output_root, dataset_name, subject_name)
    os.makedirs(subject_dir, exist_ok=True)
    meta_path = os.path.join(subject_dir, "metadata.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"subject={subject_name}\n")
        f.write(f"channel_name={channel_name}\n")
        f.write(f"target_fs={pre_cfg.target_fs}\n")
        f.write(f"epoch_sec={pre_cfg.epoch_sec}\n")
        f.write(f"wake_edge_minutes={pre_cfg.wake_edge_minutes}\n")
        f.write("bandpass_filter=none\n")
        f.write(f"n_fft={spec_cfg.n_fft}\n")
        f.write(f"win_sec={spec_cfg.win_sec}\n")
        f.write(f"overlap={spec_cfg.overlap}\n")
        f.write(f"db_min={spec_cfg.db_min}\n")
        f.write(f"db_max={spec_cfg.db_max}\n")
        f.write(f"saved_total={saved_total}\n")


def process_one_recording(
    edf_path,
    label_path,
    dataset_key,
    dataset_name,
    subject_name,
    output_root,
    pre_cfg,
    transformer,
):
    signal, channel_name = load_and_preprocess_recording(edf_path, dataset_key, pre_cfg)

    if dataset_key.lower() == "sleepedf":
        epoch_info = sleepedf_annotations_to_epochs(label_path, pre_cfg)
    else:
        epoch_info = parse_shhs_xml(label_path, pre_cfg)

    saved_total = 0
    stage_counts = {stage_id: 0 for stage_id in STAGE_NAMES.keys()}

    for onset_sec, label in epoch_info:
        epoch_signal = extract_epoch_signal(signal, onset_sec, pre_cfg)
        if epoch_signal is None:
            continue
        spec = transformer.epoch_to_spectrogram(epoch_signal)
        stage_name = STAGE_NAMES[label]
        epoch_index = stage_counts[label]
        save_epoch_spectrogram(spec, output_root, dataset_name, subject_name, stage_name, epoch_index)
        stage_counts[label] += 1
        saved_total += 1

    save_subject_metadata(output_root, dataset_name, subject_name, channel_name, saved_total, pre_cfg, transformer.cfg)
    print(f"[OK] {dataset_name}/{subject_name} | saved={saved_total} | channel={channel_name}")


def process_sleepedf(sleepedf_dir, output_root, max_subjects, pre_cfg, transformer):
    pairs = parse_sleepedf_pairs(sleepedf_dir)
    if max_subjects is not None:
        seen = set()
        limited = []
        for psg_path, hyp_path, sid, night in pairs:
            if sid not in seen:
                if len(seen) >= max_subjects:
                    break
                seen.add(sid)
            limited.append((psg_path, hyp_path, sid, night))
        pairs = limited

    for psg_path, hyp_path, sid, night in pairs:
        subject_name = f"{sid}_{night}"
        process_one_recording(
            psg_path, hyp_path, "sleepedf", "SleepEDF",
            subject_name, output_root, pre_cfg, transformer
        )


def process_shhs(shhs_edf_dir, shhs_label_dir, output_root, visit, max_subjects, pre_cfg, transformer):
    edf_files = sorted(glob.glob(os.path.join(shhs_edf_dir, "*.edf")))
    dataset_name = "SHHS1" if visit.lower() == "shhs1" else "SHHS2"
    count = 0

    for edf_path in edf_files:
        base = os.path.basename(edf_path)
        match = re.search(r"(shhs[12])-(\d+)", base.lower())
        if not match:
            continue
        file_visit, subject_id = match.group(1), match.group(2)
        if file_visit != visit.lower():
            continue
        if max_subjects is not None and count >= max_subjects:
            break

        xml_path = find_shhs_xml(shhs_label_dir, subject_id=subject_id, visit=visit.lower())
        if xml_path is None:
            print(f"[SKIP] No XML found for {visit}-{subject_id}")
            continue

        subject_name = f"{visit.lower()}-{subject_id}"
        process_one_recording(
            edf_path, xml_path, "shhs", dataset_name,
            subject_name, output_root, pre_cfg, transformer
        )
        count += 1
