import argparse
import asyncio
import os
import time
from typing import List, Tuple

import torch

from llm_instrumentation.core.streaming import StreamingSerializer


def parse_shape(shape_str: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in shape_str.split(",") if x)


async def run_stream(
    output: str,
    tensors: List[torch.Tensor],
    compression: str,
    buffer_size: int,
    workers: int,
) -> float:
    serializer = StreamingSerializer(
        buffer_size=buffer_size, num_workers=workers, compression_algo=compression
    )
    start = time.perf_counter()
    await serializer.start_streaming(output)

    for i, t in enumerate(tensors):
        await serializer.enqueue_tensor(f"layer_{i}", t)

    await serializer.stop_streaming()
    end = time.perf_counter()
    return end - start


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark optimized async streaming throughput"
    )
    parser.add_argument("--output", default="llm-instrumentation/test_output.bin")
    parser.add_argument("--num-tensors", type=int, default=256)
    parser.add_argument("--shape", default="1024,1024", help="Comma-separated dims")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "int8"],
    )
    parser.add_argument(
        "--compression", default="lz4", choices=["lz4", "zstd", "none"]
    )
    parser.add_argument("--buffer-size", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    shape = parse_shape(args.shape)

    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.int8

    tensors: List[torch.Tensor] = []
    for _ in range(args.num_tensors):
        if dtype.is_floating_point:
            t = torch.randn(*shape, dtype=dtype)
        else:
            t = (torch.rand(*shape) * 127).to(torch.int8)
        tensors.append(t)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    elapsed = asyncio.run(
        run_stream(
            output=args.output,
            tensors=tensors,
            compression=args.compression,
            buffer_size=args.buffer_size,
            workers=args.workers,
        )
    )

    size_bytes = os.path.getsize(args.output) if os.path.exists(args.output) else 0
    gbps = (size_bytes / 1e9) / elapsed if elapsed > 0 else 0.0

    print("Optimized Streaming Results")
    print(f"- Output: {args.output}")
    print(f"- Tensors: {args.num_tensors} of shape {shape} dtype={args.dtype}")
    print(f"- Compression: {args.compression}")
    print(f"- Elapsed: {elapsed:.4f} s")
    print(f"- Written: {size_bytes / (1024**2):.2f} MiB")
    print(f"- Throughput: {gbps:.3f} GB/s")


if __name__ == "__main__":
    main()
