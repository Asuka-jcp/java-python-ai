import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from tqdm import tqdm

from data.image_dataset import build_dataloaders
from models.model_factory import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_count += x.size(0)

    return total_loss / total_count, total_correct / total_count


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp: bool) -> tuple[float, float]:
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_count += x.size(0)

    return total_loss / total_count, total_correct / total_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Image classification training script")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="配置文件路径")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg["seed"])
    device = infer_device(cfg["train"]["device"])

    loaders, class_names = build_dataloaders(
        data_root=cfg["data"]["root"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    num_classes = cfg["model"]["num_classes"] or len(class_names)
    model = build_model(
        name=cfg["model"]["name"],
        num_classes=num_classes,
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["train"]["label_smoothing"])
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    use_amp = bool(cfg["train"].get("mixed_precision", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / cfg["train"]["save_name"]

    best_val_acc = 0.0
    history = []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device, use_amp
        )
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
        }
        history.append(record)
        print(record)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "config": cfg,
                    "best_val_acc": best_val_acc,
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_loss, test_acc = evaluate(model, loaders["test"], criterion, device)

    metrics = {
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }
    print("Final:", metrics)

    (save_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (save_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
