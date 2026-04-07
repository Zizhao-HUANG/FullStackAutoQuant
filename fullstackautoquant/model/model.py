"""Robust model loading with pickle backward-compatibility.

Provides two loading strategies:

1. **Preferred (state_dict)**: Construct Net() from architecture.py, then
   ``torch.load(state_dict_cpu.pt)`` -- no pickle module-path fragility.

2. **Legacy (pickle)**: ``pd.read_pickle(params.pkl)`` -- requires this shim
   file to re-export ``Net`` so that the unpickler can resolve
   ``model.model_cls`` → ``fullstackautoquant.model.model.Net``.

The ``load_model()`` function tries strategy 1 first, then falls back to 2.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from fullstackautoquant.model.architecture import (  # noqa: F401
    CausalConv1d,
    LocalSelfAttentionOverlap,
    MultiHeadSelfAttention,
    Net,
    ResidualTCNBlock,
    model_cls,
)

logger = logging.getLogger(__name__)


def load_model(
    weights_path: str | Path,
    *,
    device: str = "cpu",
    num_features: int = 6,
    num_timesteps: int = 72,
) -> tuple[Any, str]:
    """Load trained model weights with automatic strategy selection.

    Strategy priority:
      1. state_dict (.pt file) -- clean, no pickle module-path dependency
      2. Qlib pickle (.pkl file) -- legacy, requires this shim for module resolution

    Args:
        weights_path: Path to ``state_dict_cpu.pt`` or ``params.pkl``.
        device: Target device for the model (default ``"cpu"``).
        num_features: Number of input features for Net architecture.
        num_timesteps: Number of input timesteps for Net architecture.

    Returns:
        Tuple of (model_or_wrapper, strategy_used):
          - For state_dict: (nn.Module in eval mode, "state_dict")
          - For pickle: (Qlib GeneralPTNN object, "pickle")
    """
    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    # Strategy 1: state_dict loading (preferred)
    if weights_path.suffix == ".pt":
        return _load_state_dict(weights_path, device, num_features, num_timesteps), "state_dict"

    # Strategy 2: Qlib pickle loading (legacy fallback)
    if weights_path.suffix == ".pkl":
        return _load_pickle(weights_path, device), "pickle"

    # Auto-detect: try state_dict first
    try:
        return _load_state_dict(weights_path, device, num_features, num_timesteps), "state_dict"
    except Exception:
        return _load_pickle(weights_path, device), "pickle"


def _load_state_dict(
    path: Path,
    device: str,
    num_features: int,
    num_timesteps: int,
) -> Net:
    """Construct Net from architecture.py + load state_dict.

    Handles multiple formats:
    - Pure state_dict (OrderedDict of tensors) via torch.load
    - Checkpoint dict with 'model_state_dict' or 'state_dict' key
    - Full Qlib model pickle (extracts dnn_model.state_dict())
    - Raw pickle protocol 4 files (legacy .pt that are actually pickles)
    """
    import sys

    # Ensure pickle can resolve 'model.model_cls' -> this module's Net
    model_parent = str(Path(__file__).resolve().parent.parent)
    model_dir = str(Path(__file__).resolve().parent)
    paths_added = []
    for p in (model_parent, model_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
            paths_added.append(p)

    try:
        loaded = _load_raw(path, device)

        # Handle case where loaded object is a full Qlib model (not a state_dict)
        if hasattr(loaded, "dnn_model"):
            # It's a Qlib GeneralPTNN object; extract the state_dict
            dnn = loaded.dnn_model
            dnn.to(torch.device("cpu"))
            state_dict = dnn.state_dict()
        elif isinstance(loaded, dict):
            # Handle checkpoint wrappers
            if "model_state_dict" in loaded:
                state_dict = loaded["model_state_dict"]
            elif "state_dict" in loaded:
                state_dict = loaded["state_dict"]
            else:
                state_dict = loaded
        else:
            # Might be a raw nn.Module
            if hasattr(loaded, "state_dict"):
                loaded.to(torch.device("cpu"))
                state_dict = loaded.state_dict()
            else:
                raise TypeError(f"Cannot extract state_dict from {type(loaded).__name__}")

        # Auto-infer num_features from state_dict if possible
        if "input_proj.weight" in state_dict:
            num_features = state_dict["input_proj.weight"].shape[1]

        net = Net(num_features=num_features, num_timesteps=num_timesteps)
        net.load_state_dict(state_dict)
        net.eval()
        net.to(torch.device(device))
        logger.info("Loaded model via state_dict from %s (device=%s)", path, device)
        return net
    finally:
        # Clean up sys.path to avoid side effects
        for p in paths_added:
            if p in sys.path:
                sys.path.remove(p)


def _load_raw(path: Path, device: str):
    """Load a file that could be torch.save or raw pickle format."""
    import pickle

    # Detect format by reading magic bytes
    with open(path, "rb") as f:
        magic = f.read(4)

    is_raw_pickle = magic[:2] in (b"\x80\x04", b"\x80\x05", b"\x80\x03", b"\x80\x02")

    if is_raw_pickle:
        # Raw pickle format (Qlib GeneralPTNN), needs CUDA stub
        import pandas as pd

        _ensure_cuda_stub_for_loading()
        return pd.read_pickle(path)
    else:
        # torch.save format (zip or legacy torch)
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except Exception:
            _ensure_cuda_stub_for_loading()
            return torch.load(path, map_location=device, weights_only=False)


def _ensure_cuda_stub_for_loading() -> None:
    """Patch torch.load to force map_location='cpu' on CPU-only machines.

    The previous approach (replacing torch.cuda with a SimpleNamespace) causes
    libc++abi SIGABRT crashes in PyTorch >= 2.9.  Instead, we intercept
    torch.load to inject map_location='cpu'.
    """
    if torch.cuda.is_available():
        return

    if getattr(torch, "_cpu_load_patched", False):
        return

    _original_torch_load = torch.load

    def _cpu_torch_load(*args, **kwargs):
        if "map_location" not in kwargs:
            kwargs["map_location"] = "cpu"
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _cpu_torch_load  # type: ignore[assignment]
    torch._cpu_load_patched = True  # type: ignore[attr-defined]
    logger.info("Patched torch.load to force map_location='cpu' (no CUDA available)")


def _load_pickle(path: Path, device: str) -> Any:
    """Load Qlib GeneralPTNN pickle (legacy path)."""
    import pandas as pd

    model = pd.read_pickle(path)

    # Normalize device to CPU for inference
    dnn_model = getattr(model, "dnn_model", None)
    if dnn_model is not None:
        dnn_model.eval()
        dnn_model.to(torch.device(device))
    if hasattr(model, "device"):
        model.device = torch.device(device)
    if hasattr(model, "GPU"):
        try:
            model.GPU = None
        except Exception:
            model.GPU = -1
    if not getattr(model, "fitted", True):
        model.fitted = True

    logger.info("Loaded model via pickle from %s (device=%s)", path, device)
    return model
