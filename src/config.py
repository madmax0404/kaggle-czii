from __future__ import annotations
import argparse, yaml
from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"
    amp: bool = True

    train_data_dir: str = "data/train"   # folder with .npy or .zarr per volume
    val_data_dir: str = "data/val"
    output_dir: str = "outputs"

    task: str = "segmentation"           # "segmentation" or "detection-2d" (placeholder)
    in_channels: int = 1
    out_channels: int = 2                # background + foreground
    patch_size: tuple = (64, 64, 64)
    spacing: tuple | None = None

    # training
    epochs: int = 20
    batch_size: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4

    # loss/metric
    loss: str = "tversky"                # "ce" or "tversky"
    fbeta: float = 4.0                   # for f-beta metric

def load_config() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--override", type=str, nargs="*", default=[], help="key=value overrides")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        raw = yaml.safe_load(f)

    # apply overrides like train.lr=1e-4
    def set_by_path(d, path, value):
        keys = path.split(".")
        cur = d
        for k in keys[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = value

    for item in args.override:
        k, v = item.split("=", 1)
        # try to cast
        if v.isdigit():
            v = int(v)
        else:
            try:
                v = float(v)
            except:
                if v.lower() in ("true","false"):
                    v = v.lower()=="true"
        set_by_path(raw, k, v)

    # flatten nested dict into dataclass fields
    flat: Dict[str, Any] = {}
    def flatten(prefix, d):
        for k, v in d.items():
            if isinstance(v, dict):
                flatten(f"{prefix}{k}.", v)
            else:
                flat[f"{prefix}{k}"] = v
    flatten("", raw)

    # map to dataclass fields if name matches
    cfg = Config()
    for k, v in flat.items():
        attr = k.replace("train.","").replace("model.","").replace("data.","")
        if hasattr(cfg, attr):
            setattr(cfg, attr, v)
    
    return cfg