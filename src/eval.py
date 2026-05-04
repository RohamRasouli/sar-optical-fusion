"""Değerlendirme scripti — basit mAP hesabı.

Kullanım:
    python -m src.eval --checkpoint runs/final.pt --config configs/multimodal_full.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .datasets.m4_sar import collate_fn
from .losses.box_loss import bbox_iou
from .models.full_model import ModelConfig, build_model
from .train import build_dataset, load_config


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float = 0.6,
        top_k: int = 300) -> torch.Tensor:
    """Klasik NMS — çakışan kutular arasından yüksek skorluları seç."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    keep = []
    order = scores.argsort(descending=True)
    while order.numel() > 0 and len(keep) < top_k:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        ious = bbox_iou(boxes[i:i + 1].expand(order.numel() - 1, 4),
                          boxes[order[1:]], ciou=False)
        mask = ious < iou_thr
        order = order[1:][mask]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def compute_ap_per_class(detections, ground_truths, num_classes: int,
                          iou_thresholds=(0.5,)):
    """Çok basit AP hesabı — IoU eşiği başına average precision.

    detections: list of (image_id, cls, score, xyxy)
    ground_truths: dict image_id -> list of (cls, xyxy)
    """
    aps_per_iou = {}
    for iou_t in iou_thresholds:
        per_class = []
        for c in range(num_classes):
            dets_c = sorted(
                [d for d in detections if d[1] == c],
                key=lambda x: -x[2],
            )
            n_gt = sum(1 for gts in ground_truths.values()
                        for cls, _ in gts if cls == c)
            if n_gt == 0:
                continue

            tp = torch.zeros(len(dets_c))
            fp = torch.zeros(len(dets_c))
            seen = {img_id: [False] * len([1 for cls, _ in gts if cls == c])
                    for img_id, gts in ground_truths.items()}
            # Detection'larda görülebilecek ama GT'siz image_id'ler için
            for d in dets_c:
                if d[0] not in seen:
                    seen[d[0]] = []

            # Her image için class-c GT'leri
            gts_per_img = {}
            for img_id, gts in ground_truths.items():
                gts_per_img[img_id] = [box for cls, box in gts if cls == c]

            for d_i, (img_id, cls, score, xyxy) in enumerate(dets_c):
                gts = gts_per_img.get(img_id, [])
                if not gts:
                    fp[d_i] = 1
                    continue
                ious = []
                for g_box in gts:
                    iou = bbox_iou(
                        torch.tensor(xyxy).unsqueeze(0),
                        torch.tensor(g_box).unsqueeze(0),
                        ciou=False,
                    ).item()
                    ious.append(iou)
                best = max(ious)
                best_idx = ious.index(best)
                if best >= iou_t and not seen[img_id][best_idx]:
                    tp[d_i] = 1
                    seen[img_id][best_idx] = True
                else:
                    fp[d_i] = 1

            tp_cum = tp.cumsum(0)
            fp_cum = fp.cumsum(0)
            recall = tp_cum / max(n_gt, 1)
            precision = tp_cum / (tp_cum + fp_cum + 1e-9)

            # 11-point AP (basit)
            ap = 0.0
            for r in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                p = precision[recall >= r].max().item() if (recall >= r).any() else 0
                ap += p / 11
            per_class.append(ap)
        aps_per_iou[iou_t] = sum(per_class) / max(len(per_class), 1)
    return aps_per_iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conf_thr", type=float, default=0.001)
    parser.add_argument("--nms_iou", type=float, default=0.6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Hizli test icin ornek sayisi sinirla (orn: 500)")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = load_config(args.config) if args.config else ckpt["config"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
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
    model.to(device).eval()

    # Veri
    val_ds = build_dataset(cfg, split="val")
    if args.max_samples:
        val_ds = torch.utils.data.Subset(val_ds, range(min(args.max_samples, len(val_ds))))
    loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn,
                         num_workers=0)
    print(f"Val samples: {len(val_ds)}")

    img_size = m_cfg["img_size"]
    detections = []
    ground_truths = {}

    with torch.no_grad():
        for batch in loader:
            opt = batch["optical"].to(device)
            sar = batch["sar"].to(device)
            ids = batch["image_ids"]
            preds = model(opt, sar)["main"]  # (B, N, 4+nc)

            # Targets per-image
            for img_id in ids:
                ground_truths[img_id] = []
            for row in batch["labels"]:
                bidx, cls, cx, cy, w, h = row.tolist()
                bidx = int(bidx)
                if bidx >= len(ids):
                    continue
                # cx,cy,w,h normalize -> xyxy piksel
                x1 = (cx - w / 2) * img_size
                y1 = (cy - h / 2) * img_size
                x2 = (cx + w / 2) * img_size
                y2 = (cy + h / 2) * img_size
                ground_truths[ids[bidx]].append((int(cls), [x1, y1, x2, y2]))

            # Predictions
            for b in range(preds.size(0)):
                p = preds[b]                     # (N, 4+nc)
                box, scores = p[:, :4], p[:, 4:]
                # cxcywh -> xyxy
                xyxy = torch.cat([
                    box[:, :2] - box[:, 2:] / 2,
                    box[:, :2] + box[:, 2:] / 2,
                ], dim=-1)
                cls_score, cls_idx = scores.max(dim=-1)
                mask = cls_score > args.conf_thr
                xyxy = xyxy[mask]
                cls_score = cls_score[mask]
                cls_idx = cls_idx[mask]

                # NMS per-class
                keep_idx = []
                for c in cls_idx.unique():
                    cm = (cls_idx == c)
                    k = nms(xyxy[cm], cls_score[cm], iou_thr=args.nms_iou)
                    base = torch.nonzero(cm).flatten()
                    keep_idx.extend(base[k].tolist())

                for i in keep_idx:
                    detections.append((
                        ids[b],
                        int(cls_idx[i].item()),
                        float(cls_score[i].item()),
                        xyxy[i].tolist(),
                    ))

    # mAP
    iou_thrs = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    aps = compute_ap_per_class(detections, ground_truths,
                                 m_cfg["num_classes"], iou_thresholds=iou_thrs)
    print("\nEvaluation results:")
    print(f"  mAP@50    = {aps[0.5] * 100:.2f}")
    map_5095 = sum(aps.values()) / len(aps)
    print(f"  mAP@50-95 = {map_5095 * 100:.2f}")
    for t, v in aps.items():
        print(f"    IoU={t:.2f}: AP = {v * 100:.2f}")


if __name__ == "__main__":
    main()
