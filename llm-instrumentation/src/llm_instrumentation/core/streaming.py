import asyncio
import struct
from typing import Optional, List

import aiofiles
import torch

from llm_instrumentation.core.compression import TensorCompressionManager
from llm_instrumentation.memory.ring_buffer import RingBuffer


class StreamingSerializer:
    """High-throughput tensor serialization with compression."""

    def __init__(
        self,
        buffer_size: int = 128 * 1024 * 1024,  # 128MB ring buffer
        num_workers: int = 4,
        compression_algo: str = "lz4",
    ):
        """Initialize the serializer and its resources.

        - Removes reliance on a private ThreadPoolExecutor attribute.
        - Keeps worker count explicit and controlled by `num_workers`.
        """
        self.ring_buffer = RingBuffer(buffer_size)
        self.num_workers = max(1, int(num_workers))
        self.compressor = TensorCompressionManager()
        self.compressor.current_strategy = compression_algo
        self.running = False
        self.input_queue: asyncio.Queue = asyncio.Queue()
        self.writer_task: Optional[asyncio.Task] = None
        self.compression_tasks: List[asyncio.Task] = []

    async def start_streaming(self, output_path: str, resume: bool = False) -> None:
        """Start asynchronous streaming to disk."""
        if self.running:
            raise RuntimeError("Streaming is already running.")
        self.running = True

        mode = "ab" if resume else "wb"
        self.writer_task = asyncio.create_task(self._async_writer(output_path, mode))
        self.compression_tasks = [
            asyncio.create_task(self._compression_worker())
            for _ in range(self.num_workers)
        ]

    async def stop_streaming(self) -> None:
        """Stop the streaming process gracefully."""
        if not self.running:
            return

        # Signal compression workers to stop after processing remaining items
        for _ in self.compression_tasks:
            await self.input_queue.put(None)

        # Wait for all compression tasks to finish
        await asyncio.gather(*self.compression_tasks)

        # Now that no more data will be added to the ring buffer, signal writer
        self.running = False

        # Wait for the writer to finish processing the buffer
        if self.writer_task:
            await self.writer_task


    async def _async_writer(self, output_path: str, mode: str) -> None:
        """Asynchronous file writer."""
        async with aiofiles.open(output_path, mode) as f:
            while self.running or len(self.ring_buffer) > 0:
                data = await asyncio.to_thread(
                    self.ring_buffer.get, max_bytes=65536, timeout=0.1
                )
                if data:
                    await f.write(data)
            await f.flush()

    async def _compression_worker(self) -> None:
        """Worker to compress tensors from the input queue."""
        while True:
            item = await self.input_queue.get()
            if item is None:  # Sentinel value to stop the worker
                self.input_queue.task_done()
                break

            layer_name, tensor = item
            try:
                compressed_data, _ = self.compressor.compress_tensor(tensor)

                # Prepend header: name_len (H), data_len (I), name, data
                layer_name_bytes = layer_name.encode("utf-8")
                header = struct.pack("!HI", len(layer_name_bytes), len(compressed_data))
                packet = header + layer_name_bytes + compressed_data

                await asyncio.to_thread(self.ring_buffer.put, packet, timeout=1.0)
            except Exception:
                # Swallow and continue; a single bad item shouldn't kill the worker
                pass
            finally:
                self.input_queue.task_done()

    async def enqueue_tensor(self, layer_name: str, tensor: torch.Tensor) -> None:
        """Asynchronously enqueue a tensor for processing."""
        if not self.running:
            raise RuntimeError("Streaming has not been started.")
        await self.input_queue.put((layer_name, tensor))
