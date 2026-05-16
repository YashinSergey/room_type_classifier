"""
Обучение stacking-головы: backprop на train, лучший чекпоинт по val F1
(тот же протокол, что у базовых моделей в проекте).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.ensemble_i.meta_learner import StackingMetaLearner
from models.ensemble_i.members import EnsembleMember
from models.ensemble_i.predict import forward_members
from src.metrics import calculate_accuracy, calculate_macro_f1, calculate_per_class_f1


@torch.inference_mode()
def collect_member_probs(
    members: list[EnsembleMember],
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Возвращает probs [N, M, C] и targets [N]."""
    prob_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    for images, targets in tqdm(loader, desc="cache base probs", leave=False):
        images = images.to(device)
        member_probs = forward_members(images, members)
        stacked = torch.stack(member_probs, dim=1)
        prob_chunks.append(stacked.cpu())
        target_chunks.append(targets.cpu())
    return torch.cat(prob_chunks, dim=0), torch.cat(target_chunks, dim=0)


def _eval_meta_on_cache(
    meta: StackingMetaLearner,
    val_probs: torch.Tensor,
    val_targets: torch.Tensor,
    device: torch.device,
    n_members: int,
) -> tuple[float, float]:
    meta.eval()
    with torch.inference_mode():
        logits = meta([val_probs[:, m].to(device) for m in range(n_members)])
        y_pred = logits.argmax(dim=1).cpu().tolist()
        y_true = val_targets.tolist()
        return float(calculate_macro_f1(y_true, y_pred)), float(calculate_accuracy(y_true, y_pred))


def train_stacking_meta(
    members: list[EnsembleMember],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int = 200,
    lr: float = 0.05,
    weight_decay: float = 1e-3,
    patience: int = 25,
    min_delta: float = 1e-4,
    train_batch_size: int = 64,
) -> tuple[StackingMetaLearner, dict[str, Any]]:
    """
    1. Кэш softmax базовых моделей на train и val.
    2. Обучение meta только на train.
    3. Каждую эпоху — val macro F1; сохраняем веса с лучшим val F1.
  """
    n_members = len(members)
    n_classes = members[0].num_classes

    print("caching base model probs on train...", flush=True)
    train_probs, train_targets = collect_member_probs(members, train_loader, device)
    print("caching base model probs on val...", flush=True)
    val_probs, val_targets = collect_member_probs(members, val_loader, device)

    meta = StackingMetaLearner(n_members, n_classes).to(device)
    optimizer = torch.optim.AdamW(meta.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_val_acc = 0.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0

    n_train = train_probs.shape[0]
    indices = torch.arange(n_train)

    for epoch in range(1, epochs + 1):
        perm = indices[torch.randperm(n_train)]
        meta.train()
        epoch_loss = 0.0
        for start in range(0, n_train, train_batch_size):
            idx = perm[start : start + train_batch_size]
            batch_probs = [train_probs[idx, m].to(device) for m in range(n_members)]
            targets = train_targets[idx].to(device)
            logits = meta(batch_probs)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(idx)

        val_f1, val_acc = _eval_meta_on_cache(meta, val_probs, val_targets, device, n_members)

        if val_f1 > best_val_f1 + min_delta:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in meta.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch % 20 == 0 or epoch == 1 or val_f1 >= best_val_f1 - 1e-9:
            print(
                f"meta epoch {epoch}: train_loss={epoch_loss / n_train:.4f} "
                f"val_f1={val_f1:.4f} val_acc={val_acc:.4f} best_val_f1={best_val_f1:.4f}",
                flush=True,
            )
        if stale >= patience:
            print(f"early stop meta at epoch {epoch}, best_epoch={best_epoch}", flush=True)
            break

    if best_state is not None:
        meta.load_state_dict(best_state)

    meta.eval()
    with torch.inference_mode():
        logits = meta([val_probs[:, m].to(device) for m in range(n_members)])
        y_pred = logits.argmax(dim=1).cpu().tolist()
        y_true = val_targets.tolist()
        per_class = calculate_per_class_f1(y_true, y_pred, n_classes)

    info: dict[str, Any] = {
        "n_members": n_members,
        "n_classes": n_classes,
        "member_keys": [m.key for m in members],
        "best_macro_f1": float(best_val_f1),
        "best_accuracy": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "epochs_trained": epoch,
        "lr": lr,
        "weight_decay": weight_decay,
        "train_samples": int(n_train),
        "val_samples": int(val_probs.shape[0]),
        "protocol": "train_backprop_val_checkpoint",
        "per_class_metrics": per_class,
    }
    return meta, info
