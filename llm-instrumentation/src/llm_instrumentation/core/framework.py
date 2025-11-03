import asyncio
import threading
from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch

from .checkpoint import CheckpointManager
from .config import InstrumentationConfig
from .hooks import HookConfig, HookGranularity, OptimizedHookManager, TensorProcessor
from .streaming import StreamingSerializer


class _TokenBoundaryTracker:
    """Tracks token boundaries during generation without impacting streaming pipeline.

    This tracker operates outside the critical compression/streaming path to avoid
    performance overhead. Token metadata is stored in memory and flushed to JSON
    at the end of capture.
    """

    def __init__(self, checkpoint_manager: Optional[CheckpointManager] = None):
        self.tokens: list = []
        self.start_time: float = 0.0
        self._checkpoint_manager = checkpoint_manager
        self._resume_count = 0
        self._resumed = False

    def record_token(self, token_id: int, token_text: str, position: int = None):
        """Record a generated token with its metadata.

        Args:
            token_id: The token ID from the vocabulary
            token_text: The decoded token text
            position: Optional sequence position (auto-computed if None)
        """
        if position is None:
            position = len(self.tokens)

        token_payload = {
            "token_index": position,
            "token_id": int(token_id),
            "token_text": token_text,
        }
        self.tokens.append(token_payload)

        if self._checkpoint_manager and self._checkpoint_manager.has_interval():
            self._checkpoint_manager.maybe_checkpoint(
                self.tokens, self._build_checkpoint_metadata()
            )

    def save(self, path: str):
        """Save token metadata to JSON file."""
        import json
        import time

        capture_duration = time.time() - self.start_time if self.start_time else 0
        metadata = {
            "total_tokens": len(self.tokens),
            "tokens": self.tokens,
            "capture_duration_seconds": capture_duration,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        if self._checkpoint_manager:
            self._checkpoint_manager.finalize(
                self.tokens,
                self._build_checkpoint_metadata(
                    extra={"capture_duration_seconds": capture_duration}
                ),
            )

    def __enter__(self):
        import time

        self.start_time = time.time()

        if self._checkpoint_manager:
            state = self._checkpoint_manager.load()
            if state.tokens:
                self.tokens = list(state.tokens)
                stored_start = state.metadata.get("start_time")
                if stored_start:
                    self.start_time = stored_start
                self._resume_count = int(state.metadata.get("resume_count", 0)) + 1
                self._resumed = True
        return self

    def __exit__(self, *args):
        pass

    def _build_checkpoint_metadata(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        import time

        metadata = {
            "start_time": self.start_time,
            "resume_count": self._resume_count if self._resumed else 0,
            "updated_at": time.time(),
        }
        if extra:
            metadata.update(extra)
        return metadata


class InstrumentationFramework:
    """Main framework for instrumenting and monitoring LLM models."""

    def __init__(self, config: InstrumentationConfig):
        self.config = config
        self._models: Dict[int, Any] = {}
        self._active_captures: Dict[str, bool] = {}
        self._hook_manager: Optional[OptimizedHookManager] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._serializer: Optional[StreamingSerializer] = None

    def _ensure_event_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop and self._loop.is_running():
            return self._loop

        self._loop = asyncio.new_event_loop()

        def _run_loop(loop: asyncio.AbstractEventLoop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self._loop_thread = threading.Thread(target=_run_loop, args=(self._loop,), daemon=True)
        self._loop_thread.start()
        return self._loop

    def instrument_model(self, model: Any) -> None:
        """Register a model for instrumentation. Hooks are attached during capture."""
        model_id = id(model)
        if model_id in self._models:
            return
        self._models[model_id] = model

    @contextmanager
    def capture_activations(
        self,
        output_path: str,
        track_per_token: bool = False,
        checkpoint_interval_tokens: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        resume_from_checkpoint: bool = False,
    ):
        """Context manager for capturing model activations to a stream file.

        Args:
            output_path: Path to save the activation stream
            track_per_token: If True, enables per-token tracking and saves metadata
                            to {output_path}_tokens.json. Use for reasoning analysis.
            checkpoint_interval_tokens: If set, persist token checkpoints every N tokens.
            checkpoint_path: Optional explicit path for the checkpoint JSON file.
            resume_from_checkpoint: Resume token metadata and stream appends from an
                                    existing checkpoint.

        Yields:
            Optional[_TokenBoundaryTracker]: Tracker object if track_per_token=True,
                                             None otherwise

        Example:
            # Standard usage (no tracking)
            with framework.capture_activations("output.stream"):
                outputs = model.generate(...)

            # Per-token tracking for reasoning models
            with framework.capture_activations("output.stream", track_per_token=True) as tracker:
                for step in range(max_tokens):
                    outputs = model(input_ids)
                    next_token = outputs.logits[:, -1, :].argmax(dim=-1)
                    tracker.record_token(next_token.item(), tokenizer.decode(next_token))
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
        """
        if checkpoint_interval_tokens is not None and checkpoint_interval_tokens <= 0:
            raise ValueError("checkpoint_interval_tokens must be a positive integer")

        if (checkpoint_interval_tokens is not None or resume_from_checkpoint) and not track_per_token:
            raise ValueError(
                "Token checkpointing requires track_per_token=True to observe token boundaries."
            )

        # Prepare async event loop and serializer
        loop = self._ensure_event_loop()
        self._serializer = StreamingSerializer(
            compression_algo=self.config.compression_algorithm
        )

        # Start streaming in the background loop
        fut_start = asyncio.run_coroutine_threadsafe(
            self._serializer.start_streaming(output_path, resume=resume_from_checkpoint),
            loop,
        )
        fut_start.result()  # block until streaming is ready

        # Create token tracker if requested
        checkpoint_manager: Optional[CheckpointManager] = None
        if track_per_token:
            checkpoint_file = checkpoint_path or f"{output_path}.ckpt.json"
            checkpoint_manager = CheckpointManager(
                checkpoint_file,
                interval_tokens=checkpoint_interval_tokens,
                resume=resume_from_checkpoint,
            )

        token_tracker = (
            _TokenBoundaryTracker(checkpoint_manager) if track_per_token else None
        )
        if token_tracker:
            token_tracker.__enter__()

        # Create a TensorProcessor that enqueues tensors to the serializer
        processor = _StreamingTensorProcessor(self._serializer, loop)
        self._hook_manager = OptimizedHookManager(processor)

        # Attach hooks to all registered models
        hook_cfg = HookConfig(
            granularity=self.config.granularity
            if isinstance(self.config.granularity, HookGranularity)
            else HookGranularity.FULL_TENSOR,
            async_enabled=True,
        )
        for model in list(self._models.values()):
            self._hook_manager.attach_hooks(model, hook_cfg)

        self._active_captures[output_path] = True
        try:
            yield token_tracker
        finally:
            # Detach hooks first to stop new enqueueing
            if self._hook_manager:
                self._hook_manager.detach_hooks()

            # Stop streaming and wait for drain
            if self._serializer:
                fut_stop = asyncio.run_coroutine_threadsafe(
                    self._serializer.stop_streaming(), loop
                )
                fut_stop.result()

        self._active_captures[output_path] = False

        # Save token metadata if tracking was enabled
        if token_tracker and track_per_token:
            metadata_path = output_path.replace(".stream", "_tokens.json")
            token_tracker.save(metadata_path)
            token_tracker.__exit__(None, None, None)

        # Optionally stop loop if no other captures are active
        if self._loop and self._loop_thread:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=5)
            self._loop = None
            self._loop_thread = None
            self._serializer = None

    def analyze_activations(self, stream_path: str) -> Dict[str, Any]:
        """Analyze captured activations from a stream.

        Parses the stream format described in docs/STREAM_FORMAT.md and
        returns lightweight metadata useful for downstream analysis, such as
        packet counts, per-layer payload sizes, and aggregate totals.

        Note: If token tracking was enabled during capture, load the corresponding
        *_tokens.json file to correlate packets with specific generated tokens.
        """
        import os
        import struct

        if not os.path.exists(stream_path):
            raise FileNotFoundError(stream_path)

        header_fmt = "!HI"
        header_size = struct.calcsize(header_fmt)

        total_packets = 0
        total_bytes = 0
        per_layer: Dict[str, Dict[str, Any]] = {}

        with open(stream_path, "rb") as f:
            while True:
                header = f.read(header_size)
                if not header:
                    break
                if len(header) != header_size:
                    # Truncated header; stop parsing
                    break
                name_len, data_len = struct.unpack(header_fmt, header)
                name_bytes = f.read(name_len)
                if len(name_bytes) != name_len:
                    break
                layer_name = name_bytes.decode("utf-8") if name_len else ""
                payload = f.read(data_len)
                if len(payload) != data_len:
                    break

                total_packets += 1
                total_bytes += data_len
                stats = per_layer.setdefault(
                    layer_name, {"count": 0, "bytes": 0}
                )
                stats["count"] += 1
                stats["bytes"] += data_len

        return {
            "stream_path": stream_path,
            "compression": self.config.compression_algorithm,
            "packets": total_packets,
            "total_compressed_bytes": total_bytes,
            "per_layer": per_layer,
        }


class _StreamingTensorProcessor(TensorProcessor):
    """Bridges hook-captured tensors to the async StreamingSerializer."""

    def __init__(self, serializer: StreamingSerializer, loop: asyncio.AbstractEventLoop):
        self._serializer = serializer
        self._loop = loop

    def process(self, tensor: torch.Tensor, metadata: Dict[str, Any]) -> bytes:
        layer_name = metadata.get("layer_name", "")
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._serializer.enqueue_tensor(layer_name, tensor), self._loop
            )
            # We don't block on completion here to keep hooks lightweight
        except RuntimeError:
            # If loop/serializer is not available, drop silently
            pass
        return b""


def analyze_activations_with_tokens(stream_path: str, framework: InstrumentationFramework) -> Dict[str, Any]:
    """Analyze activations and load associated token metadata if available.

    Args:
        stream_path: Path to the .stream file
        framework: InstrumentationFramework instance used for capture

    Returns:
        Dictionary with activation analysis + 'token_metadata' key if available
    """
    import json
    import os

    analysis = framework.analyze_activations(stream_path)

    metadata_path = stream_path.replace(".stream", "_tokens.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            token_metadata = json.load(f)

        analysis["token_metadata"] = token_metadata

        num_tokens = token_metadata["total_tokens"]
        if num_tokens > 0 and analysis["packets"] > 0:
            analysis["packets_per_token"] = analysis["packets"] / num_tokens
            analysis["bytes_per_token"] = analysis["total_compressed_bytes"] / num_tokens

    return analysis
