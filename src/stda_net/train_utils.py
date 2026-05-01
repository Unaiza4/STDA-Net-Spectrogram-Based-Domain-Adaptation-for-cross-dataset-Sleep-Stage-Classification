import random

import numpy as np
import torch
import torch.nn.functional as F

from .constants import NUM_CLASSES
from .data import build_all_splits, read_subject_list
from .metrics import safe_div
from .models import compute_da_lambda


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_device(gpu=0):
    return torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")


def build_splits_from_args(args):
    return build_all_splits(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        max_source_subjects=getattr(args, "max_source_subjects", None),
        max_target_subjects=getattr(args, "max_target_subjects", None),
        source_subject_ids=read_subject_list(getattr(args, "source_subject_list", None)),
        target_subject_ids=read_subject_list(getattr(args, "target_subject_list", None)),
        source_train_ratio=args.source_train_ratio,
        source_val_ratio=args.source_val_ratio,
        source_test_ratio=args.source_test_ratio,
        target_train_ratio=args.target_train_ratio,
        target_val_ratio=args.target_val_ratio,
        target_test_ratio=args.target_test_ratio,
        trim_wake_ratio=args.trim_wake_ratio,
        split_seed=args.split_seed,
    )


def train_sequence_epoch(
    model,
    discriminator,
    source_loader,
    target_loader,
    optimizer,
    criterion,
    device,
    epoch,
    total_epochs,
    aux_weight=0.5,
    max_grad_norm=1.0,
    use_dann=True,
):
    model.train()
    if discriminator is not None:
        discriminator.train()

    src_iter = iter(source_loader)
    tgt_iter = iter(target_loader)
    n_batches = min(len(source_loader), len(target_loader))

    total_loss = 0.0
    total_src_correct = 0
    total_src_count = 0

    for _ in range(n_batches):
        try:
            s_x, s_y, _, _, _ = next(src_iter)
        except StopIteration:
            src_iter = iter(source_loader)
            s_x, s_y, _, _, _ = next(src_iter)

        try:
            t_x, _, _, _, _ = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(target_loader)
            t_x, _, _, _, _ = next(tgt_iter)

        s_x = s_x.to(device)
        s_y = s_y.to(device)
        t_x = t_x.to(device)

        s_feat, s_aux, s_main = model(s_x)
        loss_main = criterion(s_main.reshape(-1, NUM_CLASSES), s_y.reshape(-1))
        loss_aux = criterion(s_aux.reshape(-1, NUM_CLASSES), s_y.reshape(-1))
        loss = loss_main + aux_weight * loss_aux

        if use_dann and discriminator is not None:
            t_feat, _, _ = model(t_x)
            alpha = compute_da_lambda(epoch, total_epochs)
            s_dom = discriminator(s_feat, alpha)
            t_dom = discriminator(t_feat, alpha)
            loss_da = (
                F.binary_cross_entropy_with_logits(s_dom, torch.zeros_like(s_dom))
                + F.binary_cross_entropy_with_logits(t_dom, torch.ones_like(t_dom))
            )
            loss = loss + alpha * loss_da

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = list(model.parameters()) + (list(discriminator.parameters()) if discriminator is not None else [])
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        optimizer.step()

        preds = s_main.argmax(dim=-1)
        total_src_correct += (preds.reshape(-1) == s_y.reshape(-1)).sum().item()
        total_src_count += s_y.numel()
        total_loss += loss.item()

    return {
        "loss": safe_div(total_loss, n_batches),
        "src_acc": 100.0 * safe_div(total_src_correct, total_src_count),
    }


def train_epoch_model(
    model,
    discriminator,
    source_loader,
    target_loader,
    optimizer,
    criterion,
    device,
    epoch,
    total_epochs,
    aux_weight=0.5,
    max_grad_norm=1.0,
    use_aux=False,
    use_dann=False,
):
    model.train()
    if discriminator is not None:
        discriminator.train()

    if use_dann:
        n_batches = min(len(source_loader), len(target_loader))
        src_iter = iter(source_loader)
        tgt_iter = iter(target_loader)
    else:
        n_batches = len(source_loader)
        src_iter = iter(source_loader)
        tgt_iter = None

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for _ in range(n_batches):
        try:
            s_x, s_y, _ = next(src_iter)
        except StopIteration:
            src_iter = iter(source_loader)
            s_x, s_y, _ = next(src_iter)

        s_x = s_x.to(device)
        s_y = s_y.to(device)

        if use_aux:
            s_feat, s_main, s_aux = model(s_x)
            loss = criterion(s_main, s_y) + aux_weight * criterion(s_aux, s_y)
            logits = s_main
        else:
            s_feat, logits = model(s_x)
            loss = criterion(logits, s_y)

        if use_dann and discriminator is not None:
            try:
                t_x, _, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(target_loader)
                t_x, _, _ = next(tgt_iter)

            t_x = t_x.to(device)
            if use_aux:
                t_feat, _, _ = model(t_x)
            else:
                t_feat, _ = model(t_x)

            alpha = compute_da_lambda(epoch, total_epochs)
            s_dom = discriminator(s_feat, alpha)
            t_dom = discriminator(t_feat, alpha)
            loss_da = (
                F.binary_cross_entropy_with_logits(s_dom, torch.zeros_like(s_dom))
                + F.binary_cross_entropy_with_logits(t_dom, torch.ones_like(t_dom))
            )
            loss = loss + alpha * loss_da

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = list(model.parameters()) + (list(discriminator.parameters()) if discriminator is not None else [])
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        total_correct += (logits.argmax(1) == s_y).sum().item()
        total_count += s_y.numel()

    return {
        "loss": safe_div(total_loss, n_batches),
        "src_acc": 100.0 * safe_div(total_correct, total_count),
    }
