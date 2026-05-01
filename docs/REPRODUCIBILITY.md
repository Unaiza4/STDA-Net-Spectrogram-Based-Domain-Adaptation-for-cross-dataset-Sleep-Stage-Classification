# Reproducibility Notes

The repository does not include raw EEG recordings, generated spectrograms, trained checkpoints, or result files.

## Data flow

```text
Raw EDF/XML files
    ↓
Preprocessing
    ↓
STFT spectrogram generation
    ↓
Subject/stage organized .npy spectrograms
    ↓
STDA-Net training
```

## Splitting

Splits are subject-wise. For Sleep-EDF, both nights from the same subject are grouped under one subject identifier.

## Sequence construction

The default sequence length is 10 and stride is 5. Sequences are constructed within each recording boundary only.

## Evaluation

For overlapping windows, logits are aggregated by unique epoch file path before calculating metrics.
