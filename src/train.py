"""Eğitim scripti.

Kullanım:
    python -m src.train --config configs/multimodal_full.yaml
    python -m src.train --config configs/base.yaml --epochs 5 --batch_size 4
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .datasets.augmentation.camo_synth import CamoSynthAugmenter, CamoSynthConfig
from .datasets.m4_sar import (
    DummyM4SARDataset,
    M4SARConfig,
    M4SARDataset,
    collate_fn,
)
from .losses.camouflage_aware import CALConfig
from .losses.detection_loss import DetectionLoss
from .models.full_model import ModelConfig, build_model
from .utils.wandb_logger import WandBLogger


def dict_merge(dct, merge_dct):
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict) and isinstance(v, dict)):
            dict_merge(dct[k], v)
        else:
            dct[k] = v
    return dct

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if "defaults" in cfg:
        base_cfg = {}
        base_dir = Path(path).parent
        for item in cfg["defaults"]:
            if isinstance(item, str):
                base_path = base_dir / item
                if base_path.exists():
                    with open(base_path, "r") as bf:
                        dict_merge(base_cfg, yaml.safe_load(bf))
        cfg.pop("defaults")
        dict_merge(base_cfg, cfg)
        return base_cfg
    return cfg


def build_dataset(cfg: dict, split: str = "train"):
    """Config'e göre dataset üretir; veri yoksa dummy'e döner."""
    data_cfg = cfg["data"]
    aug_cfg = data_cfg.get("augmentation", {})
    use_camo = aug_cfg.get("camo_synth", {}).get("enabled", False) and split == "train"

    camo_aug = None
    if use_camo:
        c = aug_cfg["camo_synth"]
        camo_aug = CamoSynthAugmenter(CamoSynthConfig(
            probability=c.get("probability", 0.3),
            texture_blend_alpha=tuple(c.get("texture_blend_alpha", [0.4, 0.8])),
            net_overlay_prob=c.get("net_overlay_prob", 0.5),
        ))

    m4_cfg = M4SARConfig(
        data_root=data_cfg["data_root"],
        split=split,
        img_size=cfg["model"]["img_size"],
        augment=(split == "train"),
    )
    return M4SARDataset(m4_cfg, camo_synth_aug=camo_aug)


def build_optimizer(model: torch.nn.Module, cfg: dict):
    opt_cfg = cfg["training"]["optimizer"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
    )


def build_scheduler(optimizer, cfg: dict, num_steps: int):
    sch_cfg = cfg["training"]["scheduler"]
    warmup = sch_cfg.get("warmup_epochs", 3)
    epochs = cfg["training"]["epochs"]
    min_lr = sch_cfg.get("min_lr", 1e-6)
    initial_lr = cfg["training"]["optimizer"]["lr"]

    def lr_lambda(step):
        epoch = step / max(num_steps, 1)
        if epoch < warmup:
            return epoch / max(warmup, 1)
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return max(min_lr / initial_lr,
                    0.5 * (1 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, scaler,
                    device, epoch, log_interval=20, amp=True,
                    grad_clip=10.0, grad_accum=1, logger=None):
    model.train()
    total_loss = 0.0
    n_batches = 0
    t_start = time.time()

    for i, batch in enumerate(loader):
        opt = batch["optical"].to(device, non_blocking=True)
        sar = batch["sar"].to(device, non_blocking=True)
        targets = batch["labels"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp):
            out = model(opt, sar)
            loss_dict = loss_fn(out, targets, epoch=epoch)
            loss = loss_dict["total"] / max(grad_accum, 1)

        if amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % grad_accum == 0:
            if amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item() * max(grad_accum, 1)
        n_batches += 1

        if i % log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            comp = " | ".join(f"{k}={v.item():.3f}" for k, v in loss_dict.items()
                                if torch.is_tensor(v) and k != "total")
            print(f"  [E{epoch} B{i:04d}] loss={loss.item():.3f} lr={lr:.2e} | {comp}")

        # GPU'yu soğutmak için çok kısa bir bekleme (Sadece slow_down aktifse)
        if cfg.get("training", {}).get("slow_down", False):
            time.sleep(0.1)  # 100ms bekleme GPU yükünü %98'den %60-70'lere düşürür
            if logger is not None:
                step = epoch * len(loader) + i
                logger.log({
                    f"train/{k}": v.item() for k, v in loss_dict.items()
                    if torch.is_tensor(v)
                }, step=step)
                logger.log({"train/lr": lr}, step=step)
                if "gates" in out and out["gates"]:
                    logger.log_gating(out["gates"], step=step)

    elapsed = time.time() - t_start
    return total_loss / max(n_batches, 1), elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint (.pt) dosyasından devam et")
    parser.add_argument("--slow_down", action="store_true", help="GPU'yu yormamak için yavaşlat (gece eğitimi)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no_amp", action="store_true", help="Mixed precision'ı kapat")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.device:
        cfg["project"]["device"] = args.device
    if args.output:
        cfg["logging"]["output_dir"] = args.output

    device = torch.device(cfg["project"]["device"]
                            if torch.cuda.is_available() or cfg["project"]["device"] == "cpu"
                            else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Config: {args.config}", flush=True)

    # Reproducibility
    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)

    # Veri
    print(f"[BUSY] Veri setleri (train/val) baslatiliyor...", flush=True)
    train_ds = build_dataset(cfg, split="train")
    val_ds = build_dataset(cfg, split="val")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
        num_workers=cfg["project"]["num_workers"], collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
        num_workers=cfg["project"]["num_workers"], collate_fn=collate_fn,
    )

    # Model
    m_cfg = cfg["model"]
    model_cfg = ModelConfig(
        num_classes=m_cfg["num_classes"],
        optical_channels=m_cfg["channels"]["optical"],
        sar_channels=m_cfg["channels"]["sar"],
        encoder_base_channels=32,
        encoder_depth_mult=m_cfg["encoder"]["depth_mult"],
        encoder_width_mult=m_cfg["encoder"]["width_mult"],
        feature_channels=tuple(m_cfg["encoder"]["out_channels"]),
        cmafm_num_heads=tuple(m_cfg["cmafm"]["num_heads"]),
        cmafm_window_size=m_cfg["cmafm"]["window_size"],
        cmafm_attn_drop=m_cfg["cmafm"]["attn_dropout"],
        cmafm_drop_path=m_cfg["cmafm"]["drop_path"],
        neck_out_channels=m_cfg["neck"]["out_channels"],
        head_reg_max=m_cfg["head"]["reg_max"],
    )
    model = build_model(model_cfg).to(device)

    # Loss
    loss_cfg = cfg["loss"]
    cal_cfg = None
    if loss_cfg.get("camouflage_aware", {}).get("enabled", False):
        ca = loss_cfg["camouflage_aware"]
        cal_cfg = CALConfig(
            use_focal=ca["focal"]["enabled"],
            focal_gamma_base=ca["focal"]["gamma_base"],
            focal_beta=ca["focal"]["beta"],
            focal_lambda=ca["focal"]["lambda"],
            use_boundary=ca["boundary"]["enabled"],
            boundary_lambda=ca["boundary"]["lambda"],
            use_consistency=ca["consistency"]["enabled"],
            consistency_lambda=ca["consistency"]["lambda"],
            consistency_warmup_epochs=ca["consistency"]["warmup_epochs"],
        )

    loss_fn = DetectionLoss(
        num_classes=m_cfg["num_classes"],
        reg_max=m_cfg["head"]["reg_max"],
        box_w=loss_cfg["box_weight"],
        cls_w=loss_cfg["cls_weight"],
        dfl_w=loss_cfg["dfl_weight"],
        cal_cfg=cal_cfg,
        img_size=m_cfg["img_size"],
    ).to(device)

    # Optimizer / scheduler
    optimizer = build_optimizer(model, cfg)
    num_steps = max(1, len(train_loader) // cfg["training"].get("grad_accum_steps", 1))
    scheduler = build_scheduler(optimizer, cfg, num_steps)

    # AMP
    use_amp = cfg["training"].get("amp", True) and not args.no_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Resume (Kaldığı yerden devam et)
    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        print(f"[BUSY] Checkpoint yükleniyor: {args.resume}...", flush=True)
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[OK] Epoch {start_epoch} noktasından devam ediliyor.", flush=True)
    
    # Slow down ayarını aktar
    cfg["training"]["slow_down"] = args.slow_down

    # Çıktı dizini
    out_dir = Path(cfg["logging"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # WandB logger
    use_wandb = cfg["logging"].get("use_wandb", False)
    logger = WandBLogger(
        project=cfg["logging"].get("wandb_project", "sar-optical-fusion"),
        name=Path(args.config).stem,
        config=cfg,
        enabled=use_wandb,
    )
    if logger.enabled:
        logger.watch(model, log_freq=200)

    # Eğitim döngüsü
    best_loss = float("inf")
    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        avg_loss, elapsed = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, scaler,
            device, epoch,
            log_interval=cfg["logging"]["log_interval"],
            amp=use_amp,
            grad_clip=cfg["training"]["grad_clip_norm"],
            grad_accum=cfg["training"].get("grad_accum_steps", 1),
            logger=logger,
        )
        print(f"Epoch {epoch} tamamlandı — avg_loss={avg_loss:.4f}, süre={elapsed:.1f}s")

        # Validasyon
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                opt = batch["optical"].to(device, non_blocking=True)
                sar_v = batch["sar"].to(device, non_blocking=True)
                targets = batch["labels"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(opt, sar_v)
                    vl = loss_fn(out, targets, epoch=epoch)
                val_loss += vl["total"].item()
                n_val += 1
        val_loss /= max(n_val, 1)
        print(f"  val_loss={val_loss:.4f}")

        if logger.enabled:
            logger.log({"val/loss": val_loss}, step=(epoch + 1) * len(train_loader))

        # Checkpoint kaydet
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": cfg,
            "loss": avg_loss,
        }
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  ✓ En iyi model kaydedildi (val_loss={val_loss:.4f})")

        save_every = cfg["logging"].get("save_checkpoint_every", 5)
        if (epoch + 1) % save_every == 0:
            torch.save(ckpt, out_dir / f"epoch_{epoch}.pt")

    # Final checkpoint
    torch.save(ckpt, out_dir / "final.pt")
    print(f"\nEğitim tamamlandı. Çıktı: {out_dir}")
    logger.finish()


if __name__ == "__main__":
    main()