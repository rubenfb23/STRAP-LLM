import asyncio
import threading
from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch

from .config import InstrumentationConfig
from .hooks import HookConfig, HookGranularity, OptimizedHookManager, TensorProcessor
from .streaming import StreamingSerializer


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
    def capture_activations(self, output_path: str):
        """Context manager for capturing model activations to a stream file."""
        # Prepare async event loop and serializer
        loop = self._ensure_event_loop()
        self._serializer = StreamingSerializer(
            compression_algo=self.config.compression_algorithm
        )

        # Start streaming in the background loop
        fut_start = asyncio.run_coroutine_threadsafe(
            self._serializer.start_streaming(output_path), loop
        )
        fut_start.result()  # block until streaming is ready

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
            yield
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

            # Optionally stop loop if no other captures are active
            if self._loop and self._loop_thread:
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._loop_thread.join(timeout=5)
                self._loop = None
                self._loop_thread = None
                self._serializer = None

    def analyze_activations(self, stream_path: str) -> Dict[str, Any]:
        """Analyze captured activations from a stream."""
        # TODO: Implement actual analysis logic
        return {
            "stream_path": stream_path,
            "compression": self.config.compression_algorithm,
            "status": "analyzed",
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
