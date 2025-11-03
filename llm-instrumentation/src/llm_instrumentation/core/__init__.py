from contextlib import contextmanager
from os import PathLike
from typing import Optional, Union

from .config import InstrumentationConfig
from .framework import InstrumentationFramework


@contextmanager
def capture_activations(
    model,
    output_path: Optional[Union[str, PathLike[str]]] = None,
    output_dir: str = "./captures",
    preset: Optional[str] = None,
    config: Optional[InstrumentationConfig] = None,
    auto_timestamp: bool = True,
    prefix: str = "",
    include_metadata: bool = True,
):
    """Simplified context manager for activation capture.

    Args:
        model: PyTorch model to instrument.
        output_path: Full output file path (auto-generated when None).
        output_dir: Directory for auto-generated paths (when output_path is None).
        preset: Preset name ('balanced', 'fast_capture', 'max', 'max_compression', etc.).
        config: Custom InstrumentationConfig (ignored when preset is provided).
        auto_timestamp: Append timestamp to the filename when True.
        prefix: Prefix for auto-generated filenames (default: "").
        include_metadata: Include model and preset metadata in filename when True.

    Yields:
        InstrumentationFramework: Active instrumentation framework.

    Example:
        >>> with capture_activations(model, preset="balanced"):
        ...     outputs = model.generate(inputs, max_length=100)
    """
    import os
    from datetime import datetime

    if preset:
        preset_map = {
            "balanced": InstrumentationConfig.balanced,
            "fast_capture": InstrumentationConfig.fast_capture,
            "max_compression": InstrumentationConfig.max_compression,
            "max": InstrumentationConfig.max,
            "attention_analysis": InstrumentationConfig.attention_analysis,
            "mlp_analysis": InstrumentationConfig.mlp_analysis,
        }
        if preset not in preset_map:
            raise ValueError(f"Unknown preset: {preset}. Options: {list(preset_map.keys())}")
        config = preset_map[preset]()
    elif config is None:
        config = InstrumentationConfig.balanced()

    if output_path is None:
        os.makedirs(output_dir, exist_ok=True)

        model_name = "model"
        if include_metadata:
            if hasattr(model, "config") and hasattr(model.config, "model_type"):
                model_name = model.config.model_type
            elif hasattr(model, "__class__"):
                model_name = model.__class__.__name__.lower()
            model_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in model_name)

        parts = []
        if prefix:
            parts.append(prefix)

        if include_metadata:
            parts.append(model_name)
            parts.append(preset if preset else config.compression_algorithm)

        if auto_timestamp:
            parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))

        if not parts:
            parts.append("capture")

        filename = "_".join(parts) + ".stream"
        output_path = os.path.join(output_dir, filename)

    framework = InstrumentationFramework(config)
    framework.instrument_model(model)

    try:
        with framework.capture_activations(output_path):
            yield framework
    finally:
        pass


__all__ = ["capture_activations"]
