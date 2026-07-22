"""Train G-Mixup on DrugOOD IC50 with the shared baseline protocol."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

try:
    from .augment import make_augmented_dataset
    from .data import discover_data_root, load_splits
    from .model import GMixupGIN
except ImportError:
    from augment import make_augmented_dataset
    from data import discover_data_root, load_splits
    from model import GMixupGIN


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("assay", "scaffold", "size"), default="assay")
    parser.add_argument("--subset", choices=("core", "general", "refined"), default="core")
    parser.add_argument("--data-root", type=Path, default=discover_data_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--erm-pretrain-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--augmented-ratio", type=float, default=0.25, help="Synthetic/train ratio; sweep over 0.15, 0.25, 0.5.")
    parser.add_argument("--aug-num", type=int, default=10)
    parser.add_argument("--lam-low", type=float, default=0.005)
    parser.add_argument("--lam-high", type=float, default=0.01)
    parser.add_argument("--selection-metric", choices=("accuracy", "roc_auc"), default="accuracy")
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()
    if args.epochs < 1 or args.erm_pretrain_epochs < 0:
        parser.error("--epochs must be positive and --erm-pretrain-epochs non-negative")
    if args.augmented_ratio < 0 or args.aug_num < 1:
        parser.error("--augmented-ratio must be non-negative and --aug-num positive")
    if not 0 <= args.lam_low <= args.lam_high <= 1:
        parser.error("lambda range must satisfy 0 <= low <= high <= 1")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({name}) but is unavailable")
    return device


def make_loader(dataset, args, shuffle):
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )


def binary_auc(labels, scores):
    if np.unique(labels).size < 2:
        return math.nan
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    labels_all, scores_all, predictions_all = [], [], []
    total_loss, total_count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        labels = batch.y.view(-1).long()
        logits = model(batch)
        total_loss += float(F.cross_entropy(logits, labels, reduction="sum"))
        total_count += labels.numel()
        labels_all.append(labels.cpu())
        scores_all.append(logits.softmax(-1)[:, 1].cpu())
        predictions_all.append(logits.argmax(-1).cpu())
    labels = torch.cat(labels_all).numpy()
    scores = torch.cat(scores_all).numpy()
    predictions = torch.cat(predictions_all).numpy()
    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": float((predictions == labels).mean()),
        "roc_auc": binary_auc(labels, scores),
        "count": int(total_count),
    }


def hard_label_epoch(model, loader, device, optimizer):
    model.train()
    loss_sum, count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        labels = batch.y.view(-1).long()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batch), labels)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach()) * labels.numel()
        count += labels.numel()
    return loss_sum / count


def soft_label_epoch(model, loader, device, optimizer):
    model.train()
    loss_sum, count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        targets = batch.y.view(-1, 2).float()
        optimizer.zero_grad(set_to_none=True)
        loss = -(targets * F.log_softmax(model(batch), dim=-1)).sum(-1).mean()
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach()) * targets.shape[0]
        count += targets.shape[0]
    return loss_sum / count


def train(args):
    set_seed(args.seed)
    device = resolve_device(args.device)
    stem, splits = load_splits(args.data_root, args.subset, args.domain)
    train_loader = make_loader(splits["train"], args, True)
    eval_loaders = {name: make_loader(data, args, False) for name, data in splits.items() if name != "train"}
    sample = splits["train"][0]
    node_dim = int(sample.x.shape[-1]) if sample.x.ndim > 1 else 1
    edge_dim = int(sample.edge_attr.shape[-1]) if sample.edge_attr.ndim > 1 else 1
    model = GMixupGIN(node_dim, edge_dim, args.hidden_dim, args.num_layers, args.dropout).to(device)
    output_dir = args.output_dir or Path(__file__).resolve().parent / "outputs" / f"gmixup_{stem}_seed{args.seed}_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    history = []
    print(f"method=G-Mixup dataset={stem} device={device} output={output_dir}")

    pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(1, args.erm_pretrain_epochs + 1):
        train_loss = hard_label_epoch(model, train_loader, device, pretrain_optimizer)
        val = evaluate(model, eval_loaders["ood_val"], device)
        history.append({"phase": "erm_pretrain", "epoch": epoch, "train_loss": train_loss, "ood_val": val})
        if epoch % args.log_every == 0:
            print(f"phase=erm_pretrain epoch={epoch:03d} train={train_loss:.4f} ood_val_acc={val['accuracy']:.4f} ood_val_auc={val['roc_auc']:.4f}")

    augmented, synthetic_count = make_augmented_dataset(
        splits["train"], args.augmented_ratio, args.seed, args.aug_num,
        (args.lam_low, args.lam_high),
    )
    actual_ratio = synthetic_count / len(splits["train"])
    print(f"synthetic_graphs={synthetic_count} actual_augmented_ratio={actual_ratio:.6f}")
    augmented_loader = make_loader(augmented, args, True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_value, best_epoch, stale = -math.inf, 0, 0
    for epoch in range(1, args.epochs + 1):
        train_loss = soft_label_epoch(model, augmented_loader, device, optimizer)
        val = evaluate(model, eval_loaders["ood_val"], device)
        value = val[args.selection_metric]
        history.append({"phase": "main", "epoch": epoch, "train_loss": train_loss, "ood_val": val})
        if epoch % args.log_every == 0:
            print(f"phase=main epoch={epoch:03d} train={train_loss:.4f} ood_val_acc={val['accuracy']:.4f} ood_val_auc={val['roc_auc']:.4f}")
        if value > best_value:
            best_value, best_epoch, stale = value, epoch, 0
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, best_path)
        else:
            stale += 1
            if args.patience > 0 and stale >= args.patience:
                print(f"early stopping at epoch={epoch}; best_epoch={best_epoch}")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    metrics = {name: evaluate(model, loader, device) for name, loader in eval_loaders.items()}
    summary = {
        "method": "G-Mixup", "dataset": stem, "seed": args.seed,
        "selection_metric": args.selection_metric, "best_epoch": best_epoch,
        "best_ood_val": best_value, "synthetic_graphs": synthetic_count,
        "actual_augmented_ratio": actual_ratio, "metrics": metrics,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    train(parse_args())
