import argparse
import asyncio
import os
import time
from typing import List, Tuple

import psutil
import torch

from llm_instrumentation.core.streaming import StreamingSerializer


def parse_shape(shape_str: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in shape_str.split(",") if x)


async def produce(
    serializer: StreamingSerializer, tensors: List[torch.Tensor]
) -> None:
    for i, t in enumerate(tensors):
        await serializer.enqueue_tensor(f"layer_{i}", t)


def monitor_rss(stop_flag: List[bool], interval: float = 0.05) -> List[int]:
    proc = psutil.Process()
    samples: List[int] = []
    while not stop_flag[0]:
        samples.append(proc.memory_info().rss)
        time.sleep(interval)
    samples.append(proc.memory_info().rss)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory profiling of streaming pipeline")
    parser.add_argument("--output", default="llm-instrumentation/test_output.bin")
    parser.add_argument("--num-tensors", type=int, default=512)
    parser.add_argument("--shape", default="1024,1024")
    parser.add_argument("--compression", default="lz4", choices=["lz4", "zstd", "none"])
    parser.add_argument("--buffer-size", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    shape = parse_shape(args.shape)
    tensors = [torch.randn(*shape, dtype=torch.float32) for _ in range(args.num_tensors)]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    serializer = StreamingSerializer(
        buffer_size=args.buffer_size, num_workers=args.workers, compression_algo=args.compression
    )

    stop = [False]

    async def run():
        await serializer.start_streaming(args.output)
        await produce(serializer, tensors)
        await serializer.stop_streaming()

    # Start monitor in a thread
    import threading

    rss_samples: List[int] = []

    def monitor_thread():
        nonlocal rss_samples
        rss_samples = monitor_rss(stop)

    t = threading.Thread(target=monitor_thread, daemon=True)
    t.start()
    start = time.perf_counter()
    asyncio.run(run())
    elapsed = time.perf_counter() - start
    stop[0] = True
    t.join(timeout=1.0)

    peak_rss = max(rss_samples) if rss_samples else 0
    size_bytes = os.path.getsize(args.output) if os.path.exists(args.output) else 0
    gbps = (size_bytes / 1e9) / elapsed if elapsed > 0 else 0.0

    print("Memory Profiling Results")
    print(f"- Output: {args.output}")
    print(f"- Tensors: {args.num_tensors} of shape {shape}")
    print(f"- Buffer size: {args.buffer_size} bytes workers={args.workers}")
    print(f"- Elapsed: {elapsed:.4f} s Throughput: {gbps:.3f} GB/s")
    print(f"- Peak RSS: {peak_rss / (1024**2):.2f} MiB")


if __name__ == "__main__":
    main()
