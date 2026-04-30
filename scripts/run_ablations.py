#!/usr/bin/env python3
"""Ablation deneylerini otomatik çalıştırıcı.

base.yaml'dan başlayarak farklı bileşenler kapatılır/açılır;
her ablation için ayrı eğitim koşulur ve sonuçlar CSV'ye kaydedilir.

Kullanım:
    python scripts/run_ablations.py --base configs/multimodal_full.yaml \\
        --output runs/ablations/ --epochs 30
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import time
from pathlib import Path

import yaml


# ============================================================
# Ablation Tanımları
# ============================================================

ABLATIONS = {
    # A. Tam model — referans
    "A_full": {
        "description": "Tam model: 3-aşamalı CMAFM + tam CAL",
        "overrides": {},
    },
    # B. CMAFM yok (concat baseline)
    "B_no_cmafm": {
        "description": "CMAFM kaldırıldı, basit concat",
        "overrides": {
            "model.cmafm.enabled": False,
            "use_baseline": "concat",
        },
    },
    # C. Tek-aşamalı CMAFM (sadece 1/16'da)
    "C_single_scale_cmafm": {
        "description": "Sadece 1/16 ölçeğinde CMAFM",
        "overrides": {
            "use_baseline": "single_attn",
        },
    },
    # D. Tek-yönlü cross-attention
    "D_unidirectional_attn": {
        "description": "Sadece optic→sar yönlü attention",
        "overrides": {
            "model.cmafm.bidirectional": False,
        },
    },
    # E. Gating yok (basit toplama)
    "E_no_gating": {
        "description": "Sigmoid gating yok, F_opt + F_o2s direkt toplanıyor",
        "overrides": {
            "model.cmafm.use_gating": False,
        },
    },
    # F. CamouflageAware Loss tamamen kapalı
    "F_no_cal": {
        "description": "CAL kapalı, sadece YOLOv8 default loss",
        "overrides": {
            "loss.camouflage_aware.enabled": False,
        },
    },
    # G. CAL'dan boundary loss çıkar
    "G_no_boundary": {
        "description": "Boundary-Aware kayıp çıkarıldı",
        "overrides": {
            "loss.camouflage_aware.boundary.enabled": False,
        },
    },
    # H. CAL'dan consistency loss çıkar
    "H_no_consistency": {
        "description": "Cross-modal consistency kayıp çıkarıldı",
        "overrides": {
            "loss.camouflage_aware.consistency.enabled": False,
            "model.aux_heads": False,
        },
    },
    # I. SAR pre-training yok (random init)
    "I_no_sar_pretrain": {
        "description": "SAR backbone random init (pre-training yok)",
        "overrides": {
            "model.encoder.sar_pretrained": None,
        },
    },
    # J. Sentetik kamuflaj augmentation kaldır
    "J_no_camo_aug": {
        "description": "Sentetik kamuflaj augmentation kapalı",
        "overrides": {
            "data.augmentation.camo_synth.enabled": False,
        },
    },
}


def deep_set(cfg: dict, dotted_key: str, value):
    """'a.b.c' anahtarını cfg sözlüğüne ata."""
    parts = dotted_key.split(".")
    d = cfg
    for p in parts[:-1]:
        if p not in d:
            d[p] = {}
        d = d[p]
    d[parts[-1]] = value


def make_ablation_config(base_cfg: dict, overrides: dict, name: str) -> dict:
    """Base config'in kopyasını al, override'ları uygula."""
    cfg = copy.deepcopy(base_cfg)
    for k, v in overrides.items():
        if k.startswith("use_baseline"):
            cfg["model"]["use_baseline"] = v  # train.py gerekirse handle eder
        else:
            deep_set(cfg, k, v)
    cfg["__ablation_name"] = name
    return cfg


def run_one(name: str, cfg_path: Path, output_dir: Path, epochs: int,
             dry_run: bool = False) -> dict:
    """Bir ablation eğitimini koş ve sonuçları döndür."""
    log_path = output_dir / f"{name}.log"
    cmd = [
        "python", "-m", "src.train",
        "--config", str(cfg_path),
        "--epochs", str(epochs),
        "--output", str(output_dir / name),
    ]
    print(f"\n{'='*60}")
    print(f"Ablation {name}")
    print(f"  config: {cfg_path}")
    print(f"  epochs: {epochs}")
    print(f"{'='*60}\n")

    if dry_run:
        print(f"[DRY] {' '.join(cmd)}")
        return {"name": name, "status": "dry_run"}

    t_start = time.time()
    try:
        with open(log_path, "w") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                      timeout=24 * 3600)
        elapsed = time.time() - t_start
        status = "ok" if result.returncode == 0 else f"err({result.returncode})"
    except subprocess.TimeoutExpired:
        status = "timeout"
        elapsed = time.time() - t_start
    except Exception as e:
        status = f"exc:{e}"
        elapsed = time.time() - t_start

    # Eval koş
    ckpt = output_dir / name / "final.pt"
    map50 = map5095 = -1
    if ckpt.exists():
        eval_cmd = [
            "python", "-m", "src.eval",
            "--checkpoint", str(ckpt),
            "--config", str(cfg_path),
        ]
        eval_log = output_dir / f"{name}_eval.log"
        try:
            with open(eval_log, "w") as f:
                subprocess.run(eval_cmd, stdout=f, stderr=subprocess.STDOUT,
                                 timeout=2 * 3600)
            # Log'tan mAP'i ayıkla
            with open(eval_log, "r") as f:
                for line in f:
                    if "mAP@50 " in line and "=" in line:
                        try:
                            map50 = float(line.split("=")[-1].strip())
                        except ValueError:
                            pass
                    if "mAP@50-95" in line and "=" in line:
                        try:
                            map5095 = float(line.split("=")[-1].strip())
                        except ValueError:
                            pass
        except Exception:
            pass

    return {
        "name": name,
        "status": status,
        "elapsed_sec": round(elapsed, 1),
        "mAP50": map50,
        "mAP50-95": map5095,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="configs/multimodal_full.yaml")
    parser.add_argument("--output", type=str, default="runs/ablations")
    parser.add_argument("--epochs", type=int, default=30,
                          help="Her ablation için epoch sayısı (kısa)")
    parser.add_argument("--ablations", nargs="+", default=None,
                          help="Sadece bu ablation'ları koş (boş ise hepsi)")
    parser.add_argument("--dry_run", action="store_true",
                          help="Komutları yazdır ama çalıştırma")
    args = parser.parse_args()

    base_cfg = yaml.safe_load(open(args.base, "r"))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = args.ablations or list(ABLATIONS.keys())
    results = []

    for name in selected:
        if name not in ABLATIONS:
            print(f"  ! Bilinmeyen ablation: {name}")
            continue
        ab = ABLATIONS[name]
        print(f"\n>>> {name}: {ab['description']}")

        cfg = make_ablation_config(base_cfg, ab["overrides"], name)
        cfg_path = out_dir / f"{name}.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        result = run_one(name, cfg_path, out_dir, args.epochs, dry_run=args.dry_run)
        result["description"] = ab["description"]
        results.append(result)

        # Kaydet (her ablation sonrası, kısmi durumda dahi)
        csv_path = out_dir / "results.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "name", "description", "status", "elapsed_sec", "mAP50", "mAP50-95"
            ])
            w.writeheader()
            for r in results:
                w.writerow(r)

        json_path = out_dir / "results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # Özet
    print(f"\n{'='*70}")
    print("ABLATION SONUÇLARI")
    print(f"{'='*70}")
    print(f"{'Ablation':<20s} {'mAP@50':>8s} {'mAP@50-95':>10s} {'Status':<10s}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<20s} {r['mAP50']:>8.2f} {r['mAP50-95']:>10.2f} {r['status']:<10s}")

    print(f"\nCSV: {out_dir / 'results.csv'}")
    print(f"JSON: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
