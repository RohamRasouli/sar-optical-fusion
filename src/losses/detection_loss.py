"""Birleşik Detection Loss + Task-Aligned Assigner.

YOLOv8 stili tam loss:
  L = w_box · L_CIoU + w_cls · L_BCE + w_dfl · L_DFL
    + λ_focal · L_DynamicFocal
    + λ_bound · L_BoundaryAware
    + λ_consist · L_ConsistencyLoss

Task-Aligned Assigner (TAL):
  Her GT için, alignment_metric = cls_score^α × IoU^β skorunun en yüksek
  olduğu top-K anchor'ları positive olarak seç.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.head import dist2bbox, make_anchors
from .box_loss import CIoULoss, DFLLoss, bbox_iou
from .camouflage_aware import CALConfig, CamouflageAwareLoss


# ============================================================
# Task-Aligned Assigner
# ============================================================

class TaskAlignedAssigner:
    """YOLOv8 stili task-aligned label assignment.

    Her GT için alignment skoru:
        t = s_pred^α · IoU(b_pred, b_gt)^β
    En yüksek t skoruna sahip top-K anchor positive olarak atanır.
    """

    def __init__(self, top_k: int = 13, num_classes: int = 6,
                 alpha: float = 0.5, beta: float = 6.0, eps: float = 1e-9):
        self.top_k = top_k
        self.nc = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def __call__(self,
                 pred_scores: torch.Tensor,    # (B, N, num_classes) sigmoid sonrası
                 pred_bboxes: torch.Tensor,    # (B, N, 4) cxcywh piksel
                 anchor_points: torch.Tensor,  # (N, 2) cell coords
                 gt_labels: torch.Tensor,      # (B, M_max, 1)
                 gt_bboxes: torch.Tensor,      # (B, M_max, 4) cxcywh piksel
                 mask_gt: torch.Tensor,        # (B, M_max, 1) 0/1 padding maskesi
                 ) -> dict:
        """Returns:
            target_labels: (B, N) atanan sınıf
            target_bboxes: (B, N, 4) atanan kutu (xyxy değil cxcywh)
            target_scores: (B, N, num_classes) atanan skor (alignment metric)
            fg_mask:       (B, N) positive anchor maskesi
        """
        B, N = pred_scores.shape[:2]
        M = gt_bboxes.shape[1]

        if M == 0:
            return {
                "target_labels": torch.zeros(B, N, dtype=torch.long, device=pred_scores.device),
                "target_bboxes": torch.zeros(B, N, 4, device=pred_scores.device),
                "target_scores": torch.zeros(B, N, self.nc, device=pred_scores.device),
                "fg_mask": torch.zeros(B, N, dtype=torch.bool, device=pred_scores.device),
            }

        # 1) Anchor merkezi GT kutusunun içinde mi? (in_gts)
        # gt_bboxes: cxcywh -> xyxy
        gt_xyxy = torch.cat([
            gt_bboxes[..., :2] - gt_bboxes[..., 2:] / 2,
            gt_bboxes[..., :2] + gt_bboxes[..., 2:] / 2,
        ], dim=-1)  # (B, M, 4)

        # anchor_points: (N, 2) cell coords; piksel için stride * anchor gerekli
        # ANCAK: pred_bboxes zaten piksel cinsinden geldiği için
        # anchor merkezini piksel cinsinden kullanmalıyız.
        # Bunu external olarak strides ile çarpıp ileten caller sorumluluğunda;
        # burada anchor_points'i direkt kullanıyoruz (pred_bboxes ile aynı uzayda olmalı).
        anchors = anchor_points.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)

        x1y1 = gt_xyxy[..., :2].unsqueeze(2)  # (B, M, 1, 2)
        x2y2 = gt_xyxy[..., 2:].unsqueeze(2)
        ax = anchors.unsqueeze(1)              # (B, 1, N, 2)
        in_gts = ((ax > x1y1) & (ax < x2y2)).all(dim=-1)  # (B, M, N)
        in_gts = in_gts & mask_gt.bool()  # padded GT'leri at

        # 2) Alignment metriği: cls_score^α × IoU^β
        # cls_score: GT'nin sınıfında pred_scores
        gt_cls = gt_labels.long().squeeze(-1)  # (B, M)
        # pred_scores: (B, N, C); gt_cls: (B, M); seç gt_cls'e karşılık gelen sütun
        # ölçek (B, M, N)
        cls_per_gt = pred_scores.gather(2, gt_cls.unsqueeze(-1).expand(-1, -1, N)
                                         if False else
                                         gt_cls.unsqueeze(-1).expand(-1, -1, N)
                                         .clamp(max=self.nc - 1).unsqueeze(-1)
                                         ).squeeze(-1) if False else \
            pred_scores.permute(0, 2, 1).gather(1, gt_cls.unsqueeze(-1).expand(-1, -1, N).clamp(max=self.nc - 1))
        # Yukarıdakini sadeleştir:
        # pred_scores: (B, N, C) -> permute -> (B, C, N) -> gather'a gt_cls (B, M) → çıkış (B, M, N)
        cls_per_gt = pred_scores.permute(0, 2, 1).contiguous()  # (B, C, N)
        idx = gt_cls.clamp(max=self.nc - 1).unsqueeze(-1).expand(-1, -1, N)  # (B, M, N)
        cls_per_gt = cls_per_gt.gather(1, idx)  # (B, M, N)

        # IoU: (B, M, N)
        # pred_bboxes: (B, N, 4); gt_bboxes: (B, M, 4)
        bb_pred = pred_bboxes.unsqueeze(1).expand(-1, M, -1, -1)
        bb_gt = gt_bboxes.unsqueeze(2).expand(-1, -1, N, -1)
        ious = bbox_iou(bb_pred, bb_gt, ciou=False).clamp(0)

        align = (cls_per_gt.clamp(self.eps) ** self.alpha) * (ious.clamp(self.eps) ** self.beta)
        align = align * in_gts.float()  # sadece anchor merkezi GT içinde olanlar

        # 3) Top-K anchor seç (her GT için)
        k = min(self.top_k, N)
        topk_align, topk_idx = align.topk(k=k, dim=-1)  # (B, M, k)
        # En düşük top-K eşiği üstündekilerin maskesi
        is_in_topk = torch.zeros_like(align, dtype=torch.bool)
        is_in_topk.scatter_(2, topk_idx, topk_align > 0)

        mask_pos = is_in_topk & in_gts  # (B, M, N)

        # 4) Bir anchor birden fazla GT'ye atandıysa en yüksek IoU'yu al
        fg_mask = mask_pos.any(dim=1)  # (B, N)
        # Anchor başına en iyi GT
        if mask_pos.any():
            iou_per_anchor = (ious * mask_pos.float()).max(dim=1)
            best_iou_val = iou_per_anchor.values  # (B, N)
            best_gt_idx = iou_per_anchor.indices   # (B, N)
        else:
            best_gt_idx = torch.zeros(B, N, dtype=torch.long, device=pred_scores.device)

        # 5) Hedef tensörler
        target_labels = gt_cls.gather(1, best_gt_idx)  # (B, N)
        target_bboxes = gt_bboxes.gather(1, best_gt_idx.unsqueeze(-1).expand(-1, -1, 4))  # (B, N, 4)

        # target_scores: alignment metriği, normalize edilmiş
        target_scores = torch.zeros(B, N, self.nc, device=pred_scores.device)
        # Sadece fg_mask True olan yerlerde
        if fg_mask.any():
            # Her anchor için en iyi GT'nin alignment skorunu seç
            align_per_anchor = align.gather(1, best_gt_idx.unsqueeze(1)).squeeze(1)  # (B, N)
            # Sınıf indeksinde yerleştir
            target_scores.scatter_(2, target_labels.unsqueeze(-1),
                                     align_per_anchor.unsqueeze(-1))
            target_scores = target_scores * fg_mask.unsqueeze(-1).float()

            # Normalize per-image: max alignment = 1
            max_per_img = target_scores.amax(dim=(1, 2), keepdim=True)
            target_scores = target_scores / (max_per_img + self.eps)

        return {
            "target_labels": target_labels,
            "target_bboxes": target_bboxes,
            "target_scores": target_scores,
            "fg_mask": fg_mask,
        }


# ============================================================
# Detection Loss
# ============================================================

class DetectionLoss(nn.Module):
    """YOLOv8 stili tam loss + CamouflageAware genişletme."""

    def __init__(self, num_classes: int = 6, reg_max: int = 16,
                 box_w: float = 7.5, cls_w: float = 0.5, dfl_w: float = 1.5,
                 cal_cfg: CALConfig = None,
                 strides: List[int] = (8, 16, 32),
                 img_size: int = 640,
                 use_tal: bool = True):
        super().__init__()
        self.nc = num_classes
        self.reg_max = reg_max
        self.box_w = box_w
        self.cls_w = cls_w
        self.dfl_w = dfl_w
        self.strides = list(strides)
        self.img_size = img_size

        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.ciou = CIoULoss(reduction='none')
        self.dfl = DFLLoss(reg_max=reg_max, reduction='none')
        self.assigner = TaskAlignedAssigner(top_k=13, num_classes=num_classes)

        # DFL projeksiyonu
        self.register_buffer("proj", torch.arange(reg_max + 1, dtype=torch.float32))

        self.cal_cfg = cal_cfg
        self.cal = CamouflageAwareLoss(cal_cfg, num_classes=num_classes) \
            if cal_cfg is not None else None

    def _split_outputs(self, outs: List[torch.Tensor]):
        """Head çıktısını DFL dağılımı ve cls'e ayır.

        Returns:
            pred_dist: (B, N_total, 4, reg_max+1) — DFL ham logit
            pred_cls:  (B, N_total, num_classes) — sigmoid öncesi
            anchor_points: (N_total, 2) cell coords
            stride_tensor: (N_total, 1)
        """
        pred_box_dist, pred_cls = [], []
        for o in outs:
            B = o.size(0)
            box, cls = o.split([4 * (self.reg_max + 1), self.nc], dim=1)
            pred_box_dist.append(box.view(B, 4 * (self.reg_max + 1), -1))
            pred_cls.append(cls.view(B, self.nc, -1))
        pred_box_dist = torch.cat(pred_box_dist, dim=2).transpose(1, 2)
        pred_cls = torch.cat(pred_cls, dim=2).transpose(1, 2)
        # (B, N, 4*(reg_max+1)) → (B, N, 4, reg_max+1)
        pred_box_dist = pred_box_dist.reshape(B, -1, 4, self.reg_max + 1)

        anchor_points, stride_tensor = make_anchors(outs, self.strides)
        return pred_box_dist, pred_cls, anchor_points, stride_tensor

    def _prepare_targets(self, targets: torch.Tensor, batch_size: int):
        """Collate çıktısından (M, 6) [batch_idx, cls, cx, cy, w, h] → padded batch tensörleri."""
        device = targets.device
        if targets.numel() == 0:
            return (torch.zeros(batch_size, 0, 1, device=device),
                    torch.zeros(batch_size, 0, 4, device=device),
                    torch.zeros(batch_size, 0, 1, device=device))

        max_per_img = max(1, max(int((targets[:, 0] == i).sum().item()) for i in range(batch_size)))

        gt_labels = torch.zeros(batch_size, max_per_img, 1, device=device)
        gt_bboxes = torch.zeros(batch_size, max_per_img, 4, device=device)
        mask_gt = torch.zeros(batch_size, max_per_img, 1, device=device)

        for i in range(batch_size):
            rows = targets[targets[:, 0] == i]
            n = rows.size(0)
            if n > 0:
                gt_labels[i, :n, 0] = rows[:, 1]
                gt_bboxes[i, :n] = rows[:, 2:6]  # cx, cy, w, h (norm)
                mask_gt[i, :n] = 1.0
        return gt_labels, gt_bboxes, mask_gt

    def forward(self, model_out: dict, targets: torch.Tensor,
                epoch: int = 0) -> dict:
        """Tam loss hesabı.

        Args:
            model_out: model(optical, sar) çıktısı — 'main', 'aux_opt', 'aux_sar', 'gates'
            targets: (M, 6) [batch_idx, cls, cx, cy, w, h]
            epoch: mevcut epoch (consistency warmup için)

        Returns:
            dict with 'total', 'box', 'cls', 'dfl', ve opsiyonel CAL bileşenleri
        """
        outs = model_out["main"]  # list of (B, C, H, W) — eğitim modunda
        if not isinstance(outs, list):
            # Inference modunda çağrılırsa basit sıfır dön
            return {"total": torch.tensor(0.0, device=targets.device)}

        pred_dist, pred_cls, anchor_points, stride_tensor = self._split_outputs(outs)
        B = pred_dist.size(0)

        # Pred bbox: DFL → mesafe → cxcywh (piksel)
        pred_dist_softmax = pred_dist.softmax(dim=-1)
        pred_ltrb = (pred_dist_softmax * self.proj.reshape(1, 1, 1, -1)).sum(dim=-1)  # (B, N, 4)
        pred_bboxes = dist2bbox(pred_ltrb, anchor_points, xywh=True) * stride_tensor  # piksel

        pred_scores = pred_cls.sigmoid()  # (B, N, nc)

        # Hedefleri hazırla
        gt_labels, gt_bboxes_norm, mask_gt = self._prepare_targets(targets, B)
        # Normalize → piksel
        gt_bboxes_px = gt_bboxes_norm.clone()
        gt_bboxes_px[..., 0] *= self.img_size
        gt_bboxes_px[..., 1] *= self.img_size
        gt_bboxes_px[..., 2] *= self.img_size
        gt_bboxes_px[..., 3] *= self.img_size

        # Task-Aligned Assignment
        assigned = self.assigner(
            pred_scores.detach(), pred_bboxes.detach(),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes_px, mask_gt,
        )
        fg_mask = assigned["fg_mask"]  # (B, N)
        target_bboxes = assigned["target_bboxes"]  # (B, N, 4)
        target_scores = assigned["target_scores"]  # (B, N, nc)
        num_pos = max(fg_mask.sum().item(), 1.0)

        # --- Box loss (CIoU) ---
        if fg_mask.any():
            box_loss = self.ciou(pred_bboxes[fg_mask], target_bboxes[fg_mask]).sum() / num_pos
        else:
            box_loss = torch.tensor(0.0, device=pred_cls.device)

        # --- Cls loss (BCE) ---
        cls_loss = self.bce(pred_cls, target_scores).sum() / num_pos

        # --- DFL loss ---
        if fg_mask.any():
            # Target ltrb mesafe hesapla
            target_ltrb = torch.cat([
                (anchor_points * stride_tensor) - (target_bboxes[..., :2] - target_bboxes[..., 2:] / 2),
                (target_bboxes[..., :2] + target_bboxes[..., 2:] / 2) - (anchor_points * stride_tensor),
            ], dim=-1)  # (B, N, 4)
            target_ltrb = target_ltrb / stride_tensor  # stride'a böl (DFL aralığına getir)
            fg_dist = pred_dist[fg_mask]  # (num_pos, 4, reg_max+1)
            fg_target = target_ltrb[fg_mask]  # (num_pos, 4)
            dfl_loss = self.dfl(fg_dist.reshape(-1, self.reg_max + 1),
                                fg_target.reshape(-1)).sum() / num_pos
        else:
            dfl_loss = torch.tensor(0.0, device=pred_cls.device)

        # Toplam base loss
        loss_dict = {
            "box": box_loss * self.box_w,
            "cls": cls_loss * self.cls_w,
            "dfl": dfl_loss * self.dfl_w,
        }

        # --- CamouflageAware Loss (opsiyonel) ---
        if self.cal is not None:
            aux_opt_logits = None
            if "aux_opt" in model_out and model_out["aux_opt"] is not None:
                _, opt_cls, _, _ = self._split_outputs(model_out["aux_opt"])
                aux_opt_logits = opt_cls.reshape(-1, self.nc)

            aux_sar_logits = None
            if "aux_sar" in model_out and model_out["aux_sar"] is not None:
                _, sar_cls, _, _ = self._split_outputs(model_out["aux_sar"])
                aux_sar_logits = sar_cls.reshape(-1, self.nc)

            # Hard binary targets: focal loss 0/1 bekliyor, soft alignment scores değil
            hard_cls_targets = torch.zeros_like(target_scores)
            if fg_mask.any():
                hard_cls_targets.scatter_(2, target_labels.unsqueeze(-1), 1.0)
                hard_cls_targets = hard_cls_targets * fg_mask.unsqueeze(-1).float()

            # Model belirsizliği = zorluk proxy (1 - en yüksek sınıf güveni)
            difficulty = 1.0 - pred_scores.detach().max(dim=-1, keepdim=True)[0]  # (B, N, 1)

            cal_out = self.cal(
                cls_logits=pred_cls.reshape(-1, self.nc),
                cls_targets=hard_cls_targets.reshape(-1, self.nc),
                difficulty=difficulty.reshape(-1, 1),
                aux_opt_logits=aux_opt_logits,
                aux_sar_logits=aux_sar_logits,
                epoch=epoch,
            )
            for k, v in cal_out.items():
                if k != "total" and torch.is_tensor(v):
                    loss_dict[f"cal_{k}"] = v

        loss_dict["total"] = sum(v for v in loss_dict.values() if torch.is_tensor(v))
        return loss_dict