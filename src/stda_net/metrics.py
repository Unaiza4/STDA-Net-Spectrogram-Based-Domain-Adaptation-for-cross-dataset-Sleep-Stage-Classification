import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred) * 100.0,
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0,
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0,
        "kappa": cohen_kappa_score(y_true, y_pred) * 100.0,
    }


@torch.no_grad()
def evaluate_epoch_model(model, loader, device, has_aux=False):
    model.eval()
    all_y, all_p = [], []
    for x, label, _ in loader:
        x = x.to(device)
        label = label.to(device)
        if has_aux:
            _, logits, _ = model(x)
        else:
            _, logits = model(x)
        all_p.extend(logits.argmax(1).cpu().numpy())
        all_y.extend(label.cpu().numpy())

    y_true = np.array(all_y)
    y_pred = np.array(all_p)
    out = classification_metrics(y_true, y_pred)
    out["y_true"] = y_true
    out["y_pred"] = y_pred
    return out


@torch.no_grad()
def evaluate_sequence_model(model, loader, device, criterion=None):
    model.eval()
    logit_sum = {}
    vote_count = {}
    true_labels = {}
    total_loss = 0.0
    total_batches = 0

    for x, y, _, _, paths_batch in loader:
        x = x.to(device)
        y_dev = y.to(device)
        _, _, main_logits = model(x)

        if criterion is not None:
            num_classes = main_logits.shape[-1]
            total_loss += criterion(main_logits.reshape(-1, num_classes), y_dev.reshape(-1)).item()
            total_batches += 1

        logits_np = main_logits.detach().cpu().numpy()
        y_np = y.numpy()

        for b in range(len(paths_batch)):
            for t, path in enumerate(paths_batch[b]):
                if path not in logit_sum:
                    logit_sum[path] = logits_np[b, t].copy()
                    vote_count[path] = 1
                    true_labels[path] = int(y_np[b, t])
                else:
                    logit_sum[path] += logits_np[b, t]
                    vote_count[path] += 1

    paths = sorted(logit_sum.keys())
    y_true = np.array([true_labels[p] for p in paths], dtype=np.int64)
    y_pred = np.array([int(np.argmax(logit_sum[p] / vote_count[p])) for p in paths], dtype=np.int64)

    out = classification_metrics(y_true, y_pred)
    out["loss"] = safe_div(total_loss, total_batches)
    out["y_true"] = y_true
    out["y_pred"] = y_pred
    out["n_unique_epochs"] = len(paths)
    return out


def save_history_csv(history, out_path):
    if not history:
        return
    keys = list(history[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def save_confusion_matrix(cm, class_names, out_dir, prefix="confusion_matrix"):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{prefix}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + class_names)
        for name, row in zip(class_names, cm):
            writer.writerow([name] + [int(v) for v in row])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}.png"), dpi=200)
    plt.close(fig)
