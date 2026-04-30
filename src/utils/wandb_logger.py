"""WandB (Weights & Biases) logging entegrasyonu — opsiyonel.

train.py içinden çağrılır; WandB yüklü değilse no-op olur.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class WandBLogger:
    """WandB wrapper — yüklü değilse sessizce no-op."""

    def __init__(self, project: str, name: Optional[str] = None,
                 config: Optional[Dict] = None, enabled: bool = True):
        self.enabled = enabled
        self.run = None
        if not enabled:
            return
        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(
                project=project,
                name=name,
                config=config,
                reinit=True,
            )
        except ImportError:
            print("  ! wandb yuklu degil, logging devre disi")
            self.enabled = False
        except Exception as e:
            print(f"  ! WandB init basarisiz: {e}")
            self.enabled = False

    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        if not self.enabled:
            return
        try:
            self.wandb.log(data, step=step)
        except Exception as e:
            print(f"  ! WandB log hatasi: {e}")

    def log_image(self, name: str, image, caption: str = "", step: Optional[int] = None):
        if not self.enabled:
            return
        try:
            self.wandb.log({name: self.wandb.Image(image, caption=caption)},
                            step=step)
        except Exception:
            pass

    def log_histogram(self, name: str, values, step: Optional[int] = None):
        if not self.enabled:
            return
        try:
            self.wandb.log({name: self.wandb.Histogram(values)}, step=step)
        except Exception:
            pass

    def log_gating(self, gates: list, step: Optional[int] = None):
        """CMAFM gating bilgisini histogram olarak logla."""
        if not self.enabled:
            return
        for i, (sg_o, sg_s) in enumerate(gates):
            o = sg_o.detach().cpu().flatten().numpy()
            s = sg_s.detach().cpu().flatten().numpy()
            self.log({
                f"gates/scale_{i}/sigma_opt_mean": float(o.mean()),
                f"gates/scale_{i}/sigma_sar_mean": float(s.mean()),
                f"gates/scale_{i}/sigma_opt_std": float(o.std()),
                f"gates/scale_{i}/sigma_sar_std": float(s.std()),
            }, step=step)

    def watch(self, model, log_freq: int = 100):
        if not self.enabled:
            return
        try:
            self.wandb.watch(model, log="all", log_freq=log_freq)
        except Exception:
            pass

    def finish(self):
        if not self.enabled or self.run is None:
            return
        try:
            self.wandb.finish()
        except Exception:
            pass
