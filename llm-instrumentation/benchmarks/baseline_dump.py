import argparse
import os
import struct
import time
from typing import List, Tuple

import torch

from llm_instrumentation.core.compression import TensorCompressionManager


def parse_shape(shape_str: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in shape_str.split(",") if x)


def write_packet(fh, layer_name: str, payload: bytes) -> None:
    name_bytes = layer_name.encode("utf-8")
    header = struct.pack("!HI", len(name_bytes), len(payload))
    fh.write(header)
    fh.write(name_bytes)
    fh.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline synchronous dump (no async pipeline)"
    )
    parser.add_argument("--output", default="llm-instrumentation/test_output.bin")
    parser.add_argument("--num-tensors", type=int, default=256)
    parser.add_argument("--shape", default="1024,1024")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "int8"],
    )
    parser.add_argument(
        "--compression", default="none", choices=["lz4", "zstd", "none"]
    )
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

    mgr = TensorCompressionManager()
    mgr.current_strategy = args.compression

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    start = time.perf_counter()
    with open(args.output, "wb") as fh:
        for i in range(args.num_tensors):
            if dtype.is_floating_point:
                t = torch.randn(*shape, dtype=dtype)
            else:
                t = (torch.rand(*shape) * 127).to(torch.int8)
            payload, _ratio = mgr.compress_tensor(t)
            write_packet(fh, f"layer_{i}", payload)
        fh.flush()
        os.fsync(fh.fileno())
    elapsed = time.perf_counter() - start

    size_bytes = os.path.getsize(args.output) if os.path.exists(args.output) else 0
    gbps = (size_bytes / 1e9) / elapsed if elapsed > 0 else 0.0

    print("Baseline Dump Results")
    print(f"- Output: {args.output}")
    print(f"- Tensors: {args.num_tensors} of shape {shape} dtype={args.dtype}")
    print(f"- Compression: {args.compression}")
    print(f"- Elapsed: {elapsed:.4f} s")
    print(f"- Written: {size_bytes / (1024**2):.2f} MiB")
    print(f"- Throughput: {gbps:.3f} GB/s")


if __name__ == "__main__":
    main()
