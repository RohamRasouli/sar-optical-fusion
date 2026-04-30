"""Tek görüntü çiftine inference.

Kullanım:
    python -m src.predict \\
        --optical samples/img_001.png \\
        --sar samples/img_001.npy \\
        --checkpoint runs/final.pt \\
        --output predictions.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .datasets.m4_sar import _load_optical, _load_sar
from .datasets.preprocess import (
    SARPreprocessConfig,
    preprocess_optical,
    preprocess_sar,
)
from .eval import nms
from .models.full_model import ModelConfig, build_model


def predict_single(model, optical_path: str, sar_path: str,
                    img_size: int = 640, device: str = "cuda",
                    conf_thr: float = 0.25, nms_iou: float = 0.6) -> list:
    """Tek bir optik+SAR çiftinden tahmin döndür.

    Returns:
        [{'class': int, 'score': float, 'bbox': [x1, y1, x2, y2]}, ...]
    """
    opt = _load_optical(Path(optical_path))
    sar = _load_sar(Path(sar_path))

    # Resize
    import torch.nn.functional as F
    opt = F.interpolate(opt.unsqueeze(0).float(), size=(img_size, img_size),
                         mode='bilinear', align_corners=False).squeeze(0)
    sar = F.interpolate(sar.unsqueeze(0).float(), size=(img_size, img_size),
                         mode='bilinear', align_corners=False).squeeze(0)

    opt = preprocess_optical(opt).unsqueeze(0).to(device)
    sar = preprocess_sar(sar, SARPreprocessConfig()).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        out = model(opt, sar)
        preds = out["main"][0]  # (N, 4+nc)

    box, scores = preds[:, :4], preds[:, 4:]
    cls_score, cls_idx = scores.max(dim=-1)
    mask = cls_score > conf_thr
    box = box[mask]
    cls_score = cls_score[mask]
    cls_idx = cls_idx[mask]

    # cxcywh -> xyxy
    xyxy = torch.cat([
        box[:, :2] - box[:, 2:] / 2,
        box[:, :2] + box[:, 2:] / 2,
    ], dim=-1)

    # Per-class NMS
    keep_idx = []
    for c in cls_idx.unique():
        cm = (cls_idx == c)
        k = nms(xyxy[cm], cls_score[cm], iou_thr=nms_iou)
        base = torch.nonzero(cm).flatten()
        keep_idx.extend(base[k].tolist())

    return [
        {
            "class": int(cls_idx[i].item()),
            "score": float(cls_score[i].item()),
            "bbox": xyxy[i].tolist(),
        }
        for i in keep_idx
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optical", type=str, required=True)
    parser.add_argument("--sar", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms_iou", type=float, default=0.6)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    m_cfg = cfg["model"]
    model_cfg = ModelConfig(
        num_classes=m_cfg["num_classes"],
        optical_channels=m_cfg["channels"]["optical"],
        sar_channels=m_cfg["channels"]["sar"],
        encoder_depth_mult=m_cfg["encoder"]["depth_mult"],
        encoder_width_mult=m_cfg["encoder"]["width_mult"],
        feature_channels=tuple(m_cfg["encoder"]["out_channels"]),
        cmafm_num_heads=tuple(m_cfg["cmafm"]["num_heads"]),
        cmafm_window_size=m_cfg["cmafm"]["window_size"],
        neck_out_channels=m_cfg["neck"]["out_channels"],
        head_reg_max=m_cfg["head"]["reg_max"],
    )
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device)

    detections = predict_single(
        model, args.optical, args.sar,
        img_size=m_cfg["img_size"], device=device,
        conf_thr=args.conf, nms_iou=args.nms_iou,
    )

    print(f"\n{len(detections)} hedef tespit edildi:")
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        print(f"  class={d['class']}  score={d['score']:.3f}  "
              f"bbox=[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "optical": args.optical,
                "sar": args.sar,
                "detections": detections,
            }, f, indent=2)
        print(f"\nKaydedildi: {args.output}")


if __name__ == "__main__":
    main()
