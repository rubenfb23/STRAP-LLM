import argparse
import time
from typing import Dict, Tuple

import torch

from llm_instrumentation.core.compression import TensorCompressionManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Compression speed and ratio analysis")
    parser.add_argument("--shape", default="2048,2048")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    shape = tuple(int(x) for x in args.shape.split(",") if x)
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    torch.manual_seed(args.seed)

    # Pre-generate tensors to avoid measuring RNG cost per strategy
    tensors = [torch.randn(*shape, dtype=dtype) for _ in range(args.trials)]

    results: Dict[str, Tuple[float, float]] = {}
    for algo in ("lz4", "zstd", "none"):
        mgr = TensorCompressionManager()
        mgr.current_strategy = algo
        t0 = time.perf_counter()
        total_in = 0
        total_out = 0
        for t in tensors:
            # Manager already downcasts FP32->FP16
            payload, _ = mgr.compress_tensor(t)
            total_in += t.numel() * (2 if t.dtype == torch.float16 or t.dtype == torch.bfloat16 else 4)
            total_out += len(payload)
        dt = time.perf_counter() - t0
        ratio = (total_in / total_out) if total_out else 0.0
        results[algo] = (ratio, dt)

    print("Compression Analysis")
    print(f"- Shape: {shape} dtype={args.dtype} trials={args.trials}")
    for algo, (ratio, secs) in results.items():
        print(f"- {algo}: ratio={ratio:.2f} time={secs:.4f}s")


if __name__ == "__main__":
    main()
