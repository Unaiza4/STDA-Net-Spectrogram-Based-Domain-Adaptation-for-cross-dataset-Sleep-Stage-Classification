import os
import re
from collections import defaultdict
from typing import Optional, Set

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import NUM_CLASSES, STAGE_TO_LABEL


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def extract_subject_id(folder_name: str) -> str:
    if folder_name.startswith("SC"):
        match = re.match(r"^(SC\d{3})", folder_name)
        if match:
            return match.group(1)
        return folder_name[:5]
    return folder_name


def read_subject_list(path: Optional[str]) -> Optional[Set[str]]:
    if path is None or str(path).lower() == "null":
        return None
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def list_recording_folders(root_dir):
    return sorted(
        [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith("_")],
        key=natural_key,
    )


def filter_recordings_by_subject_ids(recordings, allowed_subject_ids=None):
    if allowed_subject_ids is None:
        return recordings
    return [rec for rec in recordings if extract_subject_id(rec) in allowed_subject_ids]


def limit_recordings_by_subject(recordings, max_subjects=None):
    if max_subjects is None:
        return recordings
    seen = set()
    limited = []
    for rec in recordings:
        sid = extract_subject_id(rec)
        if sid not in seen and len(seen) < max_subjects:
            seen.add(sid)
        if sid in seen:
            limited.append(rec)
    return limited


def collect_recording_items(root_dir, recording_name):
    rec_dir = os.path.join(root_dir, recording_name)
    items = []
    for stage_name in os.listdir(rec_dir):
        stage_dir = os.path.join(rec_dir, stage_name)
        if not os.path.isdir(stage_dir):
            continue
        label = STAGE_TO_LABEL.get(stage_name)
        if label is None:
            continue
        for fname in os.listdir(stage_dir):
            if fname.endswith(".npy"):
                items.append((os.path.join(stage_dir, fname), label, recording_name))
    return sorted(items, key=lambda x: natural_key(os.path.basename(x[0])))


def build_subject_recordings(root_dir, allowed_recordings=None):
    subject_recordings = defaultdict(dict)
    recordings = list_recording_folders(root_dir)
    if allowed_recordings is not None:
        allowed = set(allowed_recordings)
        recordings = [r for r in recordings if r in allowed]
    for rec in recordings:
        sid = extract_subject_id(rec)
        items = collect_recording_items(root_dir, rec)
        if items:
            subject_recordings[sid][rec] = items
    return subject_recordings


def split_subjects(subject_recordings, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    subjects = sorted(subject_recordings.keys(), key=natural_key)
    if len(subjects) < 3:
        raise ValueError(f"Need at least 3 subjects for splitting. Found {len(subjects)}.")

    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)

    n_total = len(subjects)
    n_train = max(1, int(round(n_total * train_ratio)))
    n_val = max(1, int(round(n_total * val_ratio)))
    n_test = n_total - n_train - n_val

    if n_test < 1:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1

    while n_train + n_val + n_test > n_total:
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            n_test -= 1

    while n_train + n_val + n_test < n_total:
        n_train += 1

    train_sids = set(subjects[:n_train])
    val_sids = set(subjects[n_train:n_train + n_val])
    test_sids = set(subjects[n_train + n_val:n_train + n_val + n_test])

    overlap = (train_sids & val_sids) | (train_sids & test_sids) | (val_sids & test_sids)
    if overlap:
        raise RuntimeError(f"Subject leakage detected: {overlap}")

    return (
        {sid: subject_recordings[sid] for sid in train_sids},
        {sid: subject_recordings[sid] for sid in val_sids},
        {sid: subject_recordings[sid] for sid in test_sids},
    )


def trim_wake_within_recording(items, trim_wake_ratio=2.0, seed=42):
    rng = np.random.RandomState(seed)
    labels = np.array([label for _, label, _ in items], dtype=np.int64)
    non_wake = labels[labels != 0]
    if len(non_wake) == 0:
        return items
    counts_nw = np.bincount(non_wake, minlength=NUM_CLASSES)
    max_nw = counts_nw[1:].max()
    target_wake = int(max_nw * trim_wake_ratio)
    wake_idx = np.where(labels == 0)[0]
    non_wake_idx = np.where(labels != 0)[0]
    if len(wake_idx) <= target_wake:
        return items
    keep_w = rng.choice(wake_idx, size=target_wake, replace=False)
    keep = np.sort(np.concatenate([keep_w, non_wake_idx]))
    return [items[i] for i in keep]


def trim_source_train_wake(subject_recordings, trim_wake_ratio=2.0, seed=42):
    out = defaultdict(dict)
    for sid, rec_map in subject_recordings.items():
        for rec, items in rec_map.items():
            out[sid][rec] = trim_wake_within_recording(items, trim_wake_ratio, seed)
    return out


def make_sequences(items, seq_len=10, stride=5):
    if len(items) < seq_len:
        return []
    return [items[i:i + seq_len] for i in range(0, len(items) - seq_len + 1, stride)]


class SleepSequenceDataset(Dataset):
    def __init__(self, subject_recordings, seq_len=10, stride=5, per_image_norm=False):
        self.samples = []
        self.per_image_norm = per_image_norm

        for sid in sorted(subject_recordings.keys(), key=natural_key):
            for rec in sorted(subject_recordings[sid].keys(), key=natural_key):
                for seq in make_sequences(subject_recordings[sid][rec], seq_len, stride):
                    paths = [p for p, _, _ in seq]
                    labels = [lab for _, lab, _ in seq]
                    self.samples.append((paths, labels, sid, rec))

        flat_labels = [label for _, labels, _, _ in self.samples for label in labels]
        counts = np.bincount(np.array(flat_labels, dtype=np.int64), minlength=NUM_CLASSES) if flat_labels else np.ones(NUM_CLASSES)
        total = max(int(counts.sum()), 1)
        weights = total / (NUM_CLASSES * counts + 1e-6)
        self.class_weights = weights / weights.sum() * NUM_CLASSES

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, labels, sid, rec = self.samples[idx]
        xs = []
        for path in paths:
            x = np.load(path).astype(np.float32)
            if x.ndim != 2:
                raise ValueError(f"Expected 2D spectrogram, got {x.shape}: {path}")
            if self.per_image_norm:
                x = (x - x.mean()) / (x.std() + 1e-6)
            xs.append(torch.from_numpy(x).unsqueeze(0))
        return torch.stack(xs), torch.tensor(labels, dtype=torch.long), sid, rec, paths

    def get_class_weights_tensor(self, device="cpu"):
        return torch.tensor(self.class_weights, dtype=torch.float32, device=device)


class SleepEpochDataset(Dataset):
    def __init__(self, subject_recordings, per_image_norm=False):
        self.samples = []
        self.per_image_norm = per_image_norm
        for sid in sorted(subject_recordings.keys(), key=natural_key):
            for rec in sorted(subject_recordings[sid].keys(), key=natural_key):
                for path, label, _ in subject_recordings[sid][rec]:
                    self.samples.append((path, label, sid))

        labels = np.array([item[1] for item in self.samples], dtype=np.int64)
        counts = np.bincount(labels, minlength=NUM_CLASSES) if len(labels) else np.ones(NUM_CLASSES)
        total = max(int(counts.sum()), 1)
        weights = total / (NUM_CLASSES * counts + 1e-6)
        self.class_weights = weights / weights.sum() * NUM_CLASSES

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, sid = self.samples[idx]
        x = np.load(path).astype(np.float32)
        if self.per_image_norm:
            x = (x - x.mean()) / (x.std() + 1e-6)
        return torch.from_numpy(x).unsqueeze(0), torch.tensor(label, dtype=torch.long), sid

    def get_class_weights_tensor(self, device="cpu"):
        return torch.tensor(self.class_weights, dtype=torch.float32, device=device)


def collate_sequence_batch(batch):
    xs, ys, sids, recs, paths = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(sids), list(recs), list(paths)


def build_all_splits(
    source_dir,
    target_dir,
    max_source_subjects=None,
    max_target_subjects=None,
    source_subject_ids=None,
    target_subject_ids=None,
    source_train_ratio=0.6,
    source_val_ratio=0.2,
    source_test_ratio=0.2,
    target_train_ratio=0.6,
    target_val_ratio=0.2,
    target_test_ratio=0.2,
    trim_wake_ratio=2.0,
    split_seed=42,
):
    src_recordings = limit_recordings_by_subject(list_recording_folders(source_dir), max_source_subjects)
    tgt_recordings = limit_recordings_by_subject(list_recording_folders(target_dir), max_target_subjects)
    src_recordings = filter_recordings_by_subject_ids(src_recordings, source_subject_ids)
    tgt_recordings = filter_recordings_by_subject_ids(tgt_recordings, target_subject_ids)

    src_sr = build_subject_recordings(source_dir, src_recordings)
    tgt_sr = build_subject_recordings(target_dir, tgt_recordings)

    src_tr, src_va, src_te = split_subjects(src_sr, source_train_ratio, source_val_ratio, source_test_ratio, split_seed)
    tgt_tr, tgt_va, tgt_te = split_subjects(tgt_sr, target_train_ratio, target_val_ratio, target_test_ratio, split_seed)
    src_tr = trim_source_train_wake(src_tr, trim_wake_ratio, split_seed)

    return {
        "src_train": src_tr,
        "src_val": src_va,
        "src_test": src_te,
        "tgt_train": tgt_tr,
        "tgt_val": tgt_va,
        "tgt_test": tgt_te,
    }
