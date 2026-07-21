"""Export validated MotionBERT and MotionAGFormer checkpoints for DirectML ONNX Runtime."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
REPOSITORIES = ROOT / "external_repos"
MODELS = ROOT / "external_models"
OUTPUT = MODELS / "onnx"


def verify_export(model: torch.nn.Module, input_data: torch.Tensor, output_path: Path) -> None:
    import onnxruntime as ort

    with torch.no_grad():
        expected = model(input_data).cpu().numpy()
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: input_data.numpy()})[0]
    maximum_error = float(np.max(np.abs(expected - actual)))
    # Transformer attention paths can differ slightly across ONNX Runtime near zero.
    # 1e-3 is below one millimeter in the model's meter-scale output.
    if not np.allclose(expected, actual, rtol=1e-3, atol=1e-3):
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"{output_path.name} divergiu do checkpoint: erro maximo {maximum_error}")
    print(f"Validated {output_path.name}: max error {maximum_error:.7f}")


def export_motionbert() -> None:
    source = REPOSITORIES / "MotionBERT"
    sys.path.insert(0, str(source))
    from lib.utils.learning import load_backbone
    from lib.utils.tools import get_config

    args = get_config(str(source / "configs" / "pose3d" / "MB_ft_h36m_global_lite.yaml"))
    model = load_backbone(args).eval()
    state = torch.load(MODELS / "motionbert-lite-pose3d.bin", map_location="cpu", weights_only=False)["model_pos"]
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    input_data = torch.zeros((1, 243, 17, 3), dtype=torch.float32)
    output_path = OUTPUT / "motionbert-lite.onnx"
    torch.onnx.export(model, input_data, output_path, input_names=["poses2d"], output_names=["poses3d"], opset_version=13)
    verify_export(model, input_data, output_path)


def export_motionagformer() -> None:
    source = REPOSITORIES / "MotionAGFormer"
    sys.path.insert(0, str(source))
    from utils.learning import load_model
    from utils.tools import get_config

    args = get_config(str(source / "configs" / "h36m" / "MotionAGFormer-xsmall.yaml"))
    model = load_model(args).eval()
    state = torch.load(MODELS / "motionagformer-xs-h36m.pth.tr", map_location="cpu", weights_only=False)["model"]
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    input_data = torch.zeros((1, 27, 17, 3), dtype=torch.float32)
    output_path = OUTPUT / "motionagformer-xs.onnx"
    torch.onnx.export(model, input_data, output_path, input_names=["poses2d"], output_names=["poses3d"], opset_version=13)
    verify_export(model, input_data, output_path)


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    export_motionbert()
    export_motionagformer()