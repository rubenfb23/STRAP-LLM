import torch
import torch.cuda as cuda
from typing import Optional, Callable
import threading


class CUDAMemoryManager:
    """Optimized CUDA memory management for high-throughput streaming."""

    def __init__(
        self,
        pinned_buffer_size: int = 512 * 1024 * 1024,  # 512MB
        num_streams: int = 2,
    ):
        self.pinned_buffer_size = pinned_buffer_size
        self.streams = [cuda.Stream() for _ in range(num_streams)]
        self.current_stream = 0

        # Allocate pinned host memory
        self.pinned_buffers = [
            torch.empty(pinned_buffer_size, dtype=torch.uint8, pin_memory=True)
            for _ in range(num_streams)
        ]

        self.lock = threading.Lock()

    def async_gpu_to_host(
        self, gpu_tensor: torch.Tensor, callback: Optional[Callable] = None
    ) -> Optional[cuda.Event]:
        """Asynchronous GPU to host transfer with double buffering."""
        if not torch.cuda.is_available():
            if callback:
                callback(gpu_tensor.cpu())
            return None

        with self.lock:
            stream_idx = self.current_stream
            self.current_stream = (self.current_stream + 1) % len(self.streams)

        stream = self.streams[stream_idx]

        with cuda.stream(stream):
            # Async copy to pinned memory
            host_tensor = gpu_tensor.to(device="cpu", non_blocking=True)

            # Record completion event
            event = cuda.Event()
            event.record(stream)

            if callback:
                # Schedule callback execution after transfer
                threading.Thread(
                    target=self._execute_after_transfer,
                    args=(event, callback, host_tensor),
                ).start()

        return event

    def _execute_after_transfer(
        self, event: cuda.Event, callback: Callable, tensor: torch.Tensor
    ) -> None:
        """Execute callback after GPU transfer completes."""
        event.wait()
        callback(tensor)
