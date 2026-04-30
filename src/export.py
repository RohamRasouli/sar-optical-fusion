"""Model dışa aktarımı: ONNX, TorchScript, TensorRT-ready.

Kullanım:
    # ONNX (en yaygın)
    python -m src.export --checkpoint runs/final.pt --format onnx --output model.onnx

    # TorchScript
    python -m src.export --checkpoint runs/final.pt --format torchscript --output model.pt

    # TensorRT engine (Jetson'da çalışmalı, host'ta TensorRT yüklü olmalı)
    python -m src.export --checkpoint runs/final.pt --format tensorrt --output model.engine
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .models.full_model import ModelConfig, build_model


class ExportableModel(torch.nn.Module):
    """Inference için sade wrapper — sadece main çıktıyı döner."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, optical, sar):
        out = self.model(optical, sar)
        return out["main"]  # (B, N, 4+nc) eval modunda


def load_model_from_checkpoint(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]
    m = cfg["model"]
    model_cfg = ModelConfig(
        num_classes=m["num_classes"],
        optical_channels=m["channels"]["optical"],
        sar_channels=m["channels"]["sar"],
        encoder_depth_mult=m["encoder"]["depth_mult"],
        encoder_width_mult=m["encoder"]["width_mult"],
        feature_channels=tuple(m["encoder"]["out_channels"]),
        cmafm_num_heads=tuple(m["cmafm"]["num_heads"]),
        cmafm_window_size=m["cmafm"]["window_size"],
        neck_out_channels=m["neck"]["out_channels"],
        head_reg_max=m["head"]["reg_max"],
        aux_heads=False,  # export için aux'a gerek yok
    )
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model, cfg


def export_onnx(model, output_path: str, img_size: int = 640,
                opt_channels: int = 3, sar_channels: int = 2,
                opset: int = 17):
    """ONNX'e aktar (dynamic batch destekli)."""
    wrapped = ExportableModel(model).eval()
    dummy_opt = torch.randn(1, opt_channels, img_size, img_size)
    dummy_sar = torch.randn(1, sar_channels, img_size, img_size)

    torch.onnx.export(
        wrapped,
        (dummy_opt, dummy_sar),
        output_path,
        input_names=["optical", "sar"],
        output_names=["detections"],
        dynamic_axes={
            "optical": {0: "batch"},
            "sar": {0: "batch"},
            "detections": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"  ✓ ONNX kaydedildi: {output_path}")

    # Doğrulama: onnxruntime ile bir forward pass koş
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path,
                                       providers=["CPUExecutionProvider"])
        out = sess.run(None, {
            "optical": dummy_opt.numpy(),
            "sar": dummy_sar.numpy(),
        })
        print(f"  ✓ ONNX runtime doğrulama: çıktı shape = {out[0].shape}")
    except ImportError:
        print("  ! onnxruntime yüklü değil, doğrulama atlandı")
    except Exception as e:
        print(f"  ! ONNX doğrulama hatası: {e}")


def export_torchscript(model, output_path: str, img_size: int = 640,
                        opt_channels: int = 3, sar_channels: int = 2):
    """TorchScript'e aktar (mobile/edge için)."""
    wrapped = ExportableModel(model).eval()
    dummy_opt = torch.randn(1, opt_channels, img_size, img_size)
    dummy_sar = torch.randn(1, sar_channels, img_size, img_size)

    traced = torch.jit.trace(wrapped, (dummy_opt, dummy_sar))
    traced.save(output_path)
    print(f"  ✓ TorchScript kaydedildi: {output_path}")


def export_tensorrt_via_trtexec(onnx_path: str, output_path: str,
                                  fp16: bool = True, int8: bool = False,
                                  workspace_mb: int = 4096):
    """trtexec komut satırı aracını kullanarak ONNX -> TensorRT engine.

    Bu fonksiyon bir TensorRT-yüklü makinede çalıştırılmalıdır.
    """
    import subprocess

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        f"--workspace={workspace_mb}",
        "--minShapes=optical:1x3x640x640,sar:1x2x640x640",
        "--optShapes=optical:1x3x640x640,sar:1x2x640x640",
        "--maxShapes=optical:8x3x640x640,sar:8x2x640x640",
    ]
    if fp16:
        cmd.append("--fp16")
    if int8:
        cmd.append("--int8")

    print(f"  Çalıştırılıyor: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✓ TensorRT engine kaydedildi: {output_path}")
    except FileNotFoundError:
        print("  ! trtexec bulunamadı. TensorRT yüklü olmalı.")
    except subprocess.CalledProcessError as e:
        print(f"  ! trtexec hata kodu: {e.returncode}")
        print(e.stdout[-500:])
        print(e.stderr[-500:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--format", type=str, choices=["onnx", "torchscript", "tensorrt"],
                          default="onnx")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--fp16", action="store_true", help="TensorRT için fp16")
    parser.add_argument("--int8", action="store_true", help="TensorRT için int8")
    args = parser.parse_args()

    print(f"Yükleniyor: {args.checkpoint}")
    model, cfg = load_model_from_checkpoint(args.checkpoint)
    img_size = args.img_size or cfg["model"]["img_size"]
    opt_ch = cfg["model"]["channels"]["optical"]
    sar_ch = cfg["model"]["channels"]["sar"]

    output = args.output or f"model.{args.format}"
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    if args.format == "onnx":
        export_onnx(model, output, img_size=img_size,
                     opt_channels=opt_ch, sar_channels=sar_ch)
    elif args.format == "torchscript":
        export_torchscript(model, output, img_size=img_size,
                            opt_channels=opt_ch, sar_channels=sar_ch)
    elif args.format == "tensorrt":
        # Önce ONNX'e
        onnx_tmp = output + ".onnx"
        export_onnx(model, onnx_tmp, img_size=img_size,
                     opt_channels=opt_ch, sar_channels=sar_ch)
        export_tensorrt_via_trtexec(onnx_tmp, output, fp16=args.fp16, int8=args.int8)


if __name__ == "__main__":
    main()
