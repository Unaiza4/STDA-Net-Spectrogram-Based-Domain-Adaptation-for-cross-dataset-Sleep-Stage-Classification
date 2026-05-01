import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stda_net.config import namespace_from_config
from stda_net.constants import LABEL_TO_STAGE, NUM_CLASSES
from stda_net.data import SleepSequenceDataset, collate_sequence_batch
from stda_net.metrics import evaluate_sequence_model, save_confusion_matrix, save_history_csv
from stda_net.models import DomainDiscriminator, STDASequenceModel
from stda_net.train_utils import build_splits_from_args, make_device, set_seed, train_sequence_epoch


def run_one(args, run_id, seed):
    set_seed(seed)
    device = make_device(args.gpu)

    run_dir = os.path.join(args.output_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    splits = build_splits_from_args(args)

    src_train_ds = SleepSequenceDataset(splits["src_train"], args.seq_len, args.stride, args.per_image_norm)
    tgt_train_ds = SleepSequenceDataset(splits["tgt_train"], args.seq_len, args.stride, args.per_image_norm)
    tgt_val_ds = SleepSequenceDataset(splits["tgt_val"], args.seq_len, args.stride, args.per_image_norm)
    tgt_test_ds = SleepSequenceDataset(splits["tgt_test"], args.seq_len, args.stride, args.per_image_norm)

    source_loader = DataLoader(src_train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True, drop_last=True,
                               collate_fn=collate_sequence_batch)
    target_loader = DataLoader(tgt_train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True, drop_last=True,
                               collate_fn=collate_sequence_batch)
    val_loader = DataLoader(tgt_val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, collate_fn=collate_sequence_batch)
    test_loader = DataLoader(tgt_test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, collate_fn=collate_sequence_batch)

    model = STDASequenceModel(NUM_CLASSES, args.feature_dim, args.lstm_hidden, args.lstm_layers, args.dropout).to(device)
    discriminator = DomainDiscriminator(args.feature_dim, args.disc_dropout).to(device)

    criterion = nn.CrossEntropyLoss(weight=src_train_ds.get_class_weights_tensor(device))
    val_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(discriminator.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)

    best_score = -1e9
    best_epoch = 0
    no_improve = 0
    history = []
    ckpt_path = os.path.join(run_dir, "best_model.pth")

    print("=" * 90)
    print(f"Proposed STDA-Net | Run {run_id} | seed={seed} | device={device}")
    print("=" * 90)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_stats = train_sequence_epoch(
            model=model,
            discriminator=discriminator,
            source_loader=source_loader,
            target_loader=target_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            aux_weight=args.aux_weight,
            max_grad_norm=args.max_grad_norm,
            use_dann=True,
        )

        val_metrics = evaluate_sequence_model(model, val_loader, device, criterion=val_criterion)
        score = val_metrics[args.monitor]
        scheduler.step(score)

        row = {
            "epoch": epoch,
            "train_loss": float(train_stats["loss"]),
            "src_acc": float(train_stats["src_acc"]),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_f1_macro": float(val_metrics["f1_macro"]),
            "val_f1_weighted": float(val_metrics["f1_weighted"]),
            "val_kappa": float(val_metrics["kappa"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_sec": float(time.time() - start),
        }
        history.append(row)

        print(
            f"E{epoch:03d}/{args.epochs} | loss={row['train_loss']:.4f} | "
            f"src_acc={row['src_acc']:.2f} | val_acc={row['val_accuracy']:.2f} | "
            f"val_f1={row['val_f1_macro']:.2f} | val_kappa={row['val_kappa']:.2f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "config": vars(args),
                },
                ckpt_path,
            )
        else:
            no_improve += 1

        save_history_csv(history, os.path.join(run_dir, "history.csv"))

        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}. Best {args.monitor}={best_score:.2f} at epoch {best_epoch}.")
            break

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    discriminator.load_state_dict(ckpt["discriminator"])

    test_metrics = evaluate_sequence_model(model, test_loader, device, criterion=val_criterion)
    cm = confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"], labels=list(range(NUM_CLASSES)))
    class_names = [LABEL_TO_STAGE[i] for i in range(NUM_CLASSES)]
    report = classification_report(test_metrics["y_true"], test_metrics["y_pred"],
                                   target_names=class_names, digits=3, zero_division=0)

    save_confusion_matrix(cm, class_names, run_dir)

    results = {
        "run_id": run_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_score": float(best_score),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_f1_macro": float(test_metrics["f1_macro"]),
        "test_f1_weighted": float(test_metrics["f1_weighted"]),
        "test_kappa": float(test_metrics["kappa"]),
        "test_unique_epochs": int(test_metrics["n_unique_epochs"]),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "history": history,
        "config": vars(args),
    }

    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_proposed.yaml")
    cli = parser.parse_args()

    args = namespace_from_config(cli.config)
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for run_id in range(args.num_runs):
        all_results.append(run_one(args, run_id, args.seed + run_id * 10))

    summary = {
        "num_runs": args.num_runs,
        "acc_mean": float(np.mean([r["test_accuracy"] for r in all_results])),
        "acc_std": float(np.std([r["test_accuracy"] for r in all_results], ddof=0)),
        "f1_macro_mean": float(np.mean([r["test_f1_macro"] for r in all_results])),
        "f1_macro_std": float(np.std([r["test_f1_macro"] for r in all_results], ddof=0)),
        "kappa_mean": float(np.mean([r["test_kappa"] for r in all_results])),
        "kappa_std": float(np.std([r["test_kappa"] for r in all_results], ddof=0)),
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
