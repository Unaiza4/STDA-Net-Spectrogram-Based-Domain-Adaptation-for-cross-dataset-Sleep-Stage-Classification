# STDA-Net: SSpectrogram-Based Domain Adaptation for cross-dataset Sleep Stage Classificationn

This repository contains a clean implementation of **STDA-Net**, a spectrogram-based temporal domain adaptation framework for cross-dataset EEG sleep stage classification.

## What is included

- EEG preprocessing and STFT spectrogram generation
- Subject-wise train/validation/test splitting
- CNN-based spectrogram encoder
- Auxiliary classifier supervision
- BiLSTM temporal modeling
- Domain adversarial training with gradient reversal
- Ablation variants A1–A6

## Repository structure

```text
configs/
  generate_spectrograms.yaml
  train_proposed.yaml
  train_ablation.yaml

docs/
  REPRODUCIBILITY.md

scripts/
  generate_spectrograms.py
  make_subject_lists.py
  train_proposed.py
  train_ablation.py

src/stda_net/
  constants.py
  config.py
  preprocessing.py
  spectrograms.py
  models.py
  metrics.py
  train_utils.py


## Installation

```bash
pip install -r requirements.txt
```

## Preprocessing used by default

1. EEG channel selection
2. Sleep-stage label parsing
3. N3/N4 merge into one N3 class
4. Wake-edge trimming
5. Filtering
6. Resampling to 100 Hz
7. Epoch-level normalization
8. STFT spectrogram generation
9. dB clipping to `[-80, 0]` and normalization to `[0, 1]`

## Data format

Generated spectrograms should follow:

```text
DatasetRoot/
  SubjectID/
    W/
      epoch_0000.npy
    N1/
    N2/
    N3/
    REM/
```

## Run spectrogram generation

Edit:

```text
configs/generate_spectrograms.yaml
```

Then run:

```bash
python scripts/generate_spectrograms.py --config configs/generate_spectrograms.yaml
```

## Train proposed STDA-Net

Edit:

```text
configs/train_proposed.yaml
```

Then run:

```bash
python scripts/train_proposed.py --config configs/train_proposed.yaml
```

## Train ablation model

Edit:

```text
configs/train_ablation.yaml
```

Then run:

```bash
python scripts/train_ablation.py --config configs/train_ablation.yaml
```

## Ablation variants

```text
A1: CNN only
A2: CNN + auxiliary classifier
A3: CNN + DANN
A4: CNN + auxiliary classifier + DANN
A5: CNN + auxiliary classifier + BiLSTM
A6: CNN + auxiliary classifier + BiLSTM + DANN
```

The proposed model is A6.

Implementation Details:

The model is implemented in PyTorch and trained on an NVIDIA Quadro RTX 6000 using the Adam optimizer with a learning rate of $3 \times 10^{-4}$ and weight decay of $10^{-4}$. The learning rate is reduced by a factor of 0.5 if the validation MF1 plateaus for 3 consecutive evaluations, with a minimum value of $10^{-6}$. Training uses a batch size of 4 sequences, with 10 consecutive epochs per batch. Gradient clipping with a maximum norm of 1.0 is applied to stabilize adversarial training