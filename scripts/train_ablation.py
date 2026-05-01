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
from stda_net.data import SleepEpochDataset, SleepSequenceDataset, collate_sequence_batch
from stda_net.metrics import evaluate_epoch_model, evaluate_sequence_model, save_confusion_matrix, save_history_csv
from stda_net.models import CNNOnly, CNNAux, DomainDiscriminator, STDASequenceModel
from stda_net.train_utils import (
    build_splits_from_args,
    make_device,
    set_seed,
    train_epoch_model,
    train_sequence_epoch,
)


VARIANT_DESC = {
    "A1": "CNN only",
    "A2": "CNN + auxiliary classifier",
    "A3": "CNN + DANN",
    "A4": "CNN + auxiliary classifier + DANN",
    "A5": "CNN + auxiliary classifier + BiLSTM",
    "A6": "CNN + auxiliary classifier + BiLSTM + DANN",
}


def build_model_and_flags(args, device):
    variant = args.variant.upper()
    is_sequence = variant in ("A5", "A6")
    use_dann = variant in ("A3", "A4", "A6")
    use_aux = variant in ("A2", "A4")

    if variant == "A1":
        model = CNNOnly(NUM_CLASSES, args.feature_dim, args.dropout).to(device)
    elif variant == "A2":
        model = CNNAux(NUM_CLASSES, args.feature_dim, args.dropout).to(device)
    elif variant == "A3":
        model = CNNOnly(NUM_CLASSES, args.feature_dim, args.dropout).to(device)
    elif variant == "A4":
        model = CNNAux(NUM_CLASSES, args.feature_dim, args.dropout).to(device)
    elif variant in ("A5", "A6"):
        model = STDASequenceModel(NUM_CLASSES, args.feature_dim, args.lstm_hidden, args.lstm_layers, args.dropout).to(device)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    discriminator = DomainDiscriminator(args.feature_dim, args.disc_dropout).to(device) if use_dann else None
    return model, discriminator, is_sequence, use_dann, use_aux


def run_one(args, run_id, seed):
    set_seed(seed)
    device = make_device(args.gpu)

    run_dir = os.path.join(args.output_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    splits = build_splits_from_args(args)
    model, discriminator, is_sequence, use_dann, use_aux = build_model_and_flags(args, device)

    if is_sequence:
        src_train_ds = SleepSequenceDataset(splits["src_train"], args.seq_len, args.stride, args.per_image_norm)
        tgt_train_ds = SleepSequenceDataset(splits["tgt_train"], args.seq_len, args.stride, args.per_image_norm)
        tgt_val_ds = SleepSequenceDataset(splits["tgt_val"], args.seq_len, args.stride, args.per_image_norm)
        tgt_test_ds = SleepSequenceDataset(splits["tgt_test"], args.seq_len, args.stride, args.per_image_norm)
        collate_fn = collate_sequence_batch
        batch_size = args.batch_size
    else:
        src_train_ds = SleepEpochDataset(splits["src_train"], args.per_image_norm)
        tgt_train_ds = SleepEpochDataset(splits["tgt_train"], args.per_image_norm)
        tgt_val_ds = SleepEpochDataset(splits["tgt_val"], args.per_image_norm)
        tgt_test_ds = SleepEpochDataset(splits["tgt_test"], args.per_image_norm)
        collate_fn = None
        batch_size = args.batch_size * args.seq_len

    source_loader = DataLoader(src_train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True, drop_last=True,
                               collate_fn=collate_fn)
    target_loader = DataLoader(tgt_train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True, drop_last=True,
                               collate_fn=collate_fn)
    val_loader = DataLoader(tgt_val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(tgt_test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss(weight=src_train_ds.get_class_weights_tensor(device))
    params = list(model.parameters()) + (list(discriminator.parameters()) if discriminator is not None else [])
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)

    best_score = -1e9
    best_epoch = 0
    no_improve = 0
    history = []
    ckpt_path = os.path.join(run_dir, f"best_{args.variant}.pth")

    print("=" * 90)
    print(f"{args.variant}: {VARIANT_DESC[args.variant]} | Run {run_id} | seed={seed} | device={device}")
    print("=" * 90)

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        if is_sequence:
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
                use_dann=use_dann,
            )
            val_metrics = evaluate_sequence_model(model, val_loader, device)
        else:
            train_stats = train_epoch_model(
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
                use_aux=use_aux,
                use_dann=use_dann,
            )
            val_metrics = evaluate_epoch_model(model, val_loader, device, has_aux=use_aux)

        score = val_metrics["f1_macro"]
        scheduler.step(score)

        row = {
            "epoch": epoch,
            "train_loss": float(train_stats["loss"]),
            "src_acc": float(train_stats["src_acc"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_f1_macro": float(val_metrics["f1_macro"]),
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
            save_dict = {
                "model": model.state_dict(),
                "best_epoch": best_epoch,
                "best_score": best_score,
                "variant": args.variant,
                "config": vars(args),
            }
            if discriminator is not None:
                save_dict["discriminator"] = discriminator.state_dict()
            torch.save(save_dict, ckpt_path)
        else:
            no_improve += 1

        save_history_csv(history, os.path.join(run_dir, "history.csv"))

        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}. Best F1={best_score:.2f} at epoch {best_epoch}.")
            break

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if discriminator is not None and "discriminator" in ckpt:
        discriminator.load_state_dict(ckpt["discriminator"])

    if is_sequence:
        test_metrics = evaluate_sequence_model(model, test_loader, device)
    else:
        test_metrics = evaluate_epoch_model(model, test_loader, device, has_aux=use_aux)

    cm = confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"], labels=list(range(NUM_CLASSES)))
    class_names = [LABEL_TO_STAGE[i] for i in range(NUM_CLASSES)]
    report = classification_report(test_metrics["y_true"], test_metrics["y_pred"],
                                   target_names=class_names, digits=3, zero_division=0)

    save_confusion_matrix(cm, class_names, run_dir)

    results = {
        "variant": args.variant,
        "description": VARIANT_DESC[args.variant],
        "run_id": run_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_f1_macro": float(best_score),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_f1_macro": float(test_metrics["f1_macro"]),
        "test_kappa": float(test_metrics["kappa"]),
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
    parser.add_argument("--config", type=str, default="configs/train_ablation.yaml")
    cli = parser.parse_args()

    args = namespace_from_config(cli.config)
    args.variant = args.variant.upper()
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for run_id in range(args.num_runs):
        all_results.append(run_one(args, run_id, args.seed + run_id * 10))

    summary = {
        "variant": args.variant,
        "description": VARIANT_DESC[args.variant],
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
