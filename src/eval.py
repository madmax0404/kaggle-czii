from __future__ import annotations
import os, torch, argparse
from torch.utils.data import DataLoader
from config import load_config, Config
from datasets import CryoETSegDataset
from models.unet3d import build_unet3d
from metrics import fbeta_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/best.pt")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args, unknown = parser.parse_known_args()
    # load config with optional overrides
    import yaml
    with open(args.config,"r") as f:
        raw = yaml.safe_load(f)
    cfg = load_config()  # uses args --config if passed via CLI
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    ds = CryoETSegDataset(cfg.val_data_dir, patch_size=tuple(cfg.patch_size), spacing=cfg.spacing, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = build_unet3d(cfg.in_channels, cfg.out_channels).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    f_sum, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(imgs)
            f = fbeta_score(logits, labels, beta=cfg.fbeta, num_classes=cfg.out_channels)
            f_sum += f; n += 1
    print(f"Eval F{cfg.fbeta:.0f}: {f_sum/max(1,n):.4f}")

if __name__ == "__main__":
    main()