from __future__ import annotations
import os, time
import torch
from torch.utils.data import DataLoader
from torch import optim
from config import load_config
from utils import seed_everything, AverageMeter, save_checkpoint
from datasets import CryoETSegDataset
from models.unet3d import build_unet3d
from losses import make_loss
from metrics import fbeta_score

def main():
    cfg = load_config()
    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Datasets / loaders
    train_ds = CryoETSegDataset(cfg.train_data_dir, patch_size=tuple(cfg.patch_size), spacing=cfg.spacing, augment=True)
    val_ds   = CryoETSegDataset(cfg.val_data_dir,   patch_size=tuple(cfg.patch_size), spacing=cfg.spacing, augment=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,              shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # Model / loss / opt
    model = build_unet3d(cfg.in_channels, cfg.out_channels).to(device)
    criterion = make_loss(cfg.loss, out_channels=cfg.out_channels)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type=="cuda"))

    best_f = -1.0
    os.makedirs(cfg.output_dir, exist_ok=True)

    for epoch in range(1, cfg.epochs+1):
        model.train()
        loss_meter = AverageMeter()

        for batch in train_loader:
            imgs = batch["image"].to(device)          # (N,1,D,H,W)
            labels = batch["label"].to(device)        # (N,D,H,W)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(imgs)                  # (N,C,D,H,W)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_meter.update(loss.item(), imgs.size(0))

        # validation
        model.eval()
        f_meter = AverageMeter()
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(imgs)
                f = fbeta_score(logits, labels, beta=cfg.fbeta, num_classes=cfg.out_channels)
                f_meter.update(f, imgs.size(0))

        print(f"Epoch {epoch:03d} | train_loss={loss_meter.avg:.4f} | val_f{cfg.fbeta:.0f}={f_meter.avg:.4f}")

        # save best
        if f_meter.avg > best_f:
            best_f = f_meter.avg
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_f": best_f,
                "config": vars(cfg),
            }, os.path.join(cfg.output_dir, "best.pt"))
    print(f"Training done. Best F{cfg.fbeta:.0f}: {best_f:.4f}")

if __name__ == "__main__":
    main()