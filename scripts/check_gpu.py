#!/usr/bin/env python3
"""Verify that PyTorch can execute a small matrix operation on every visible GPU."""

from __future__ import annotations

import argparse

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-devices",
        type=int,
        default=0,
        help="Fail unless at least this many CUDA devices are available.",
    )
    parser.add_argument("--matrix-size", type=int, default=512)
    args = parser.parse_args()

    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA devices: {count}")

    if count < args.min_devices:
        raise SystemExit(
            f"Expected at least {args.min_devices} CUDA device(s), but found {count}."
        )

    for index in range(count):
        device = torch.device(f"cuda:{index}")
        generator = torch.Generator(device=device).manual_seed(2026 + index)
        value = torch.randn(
            (args.matrix_size, args.matrix_size),
            device=device,
            generator=generator,
        )
        result = value @ value.T
        torch.cuda.synchronize(device)
        print(
            f"cuda:{index}: {torch.cuda.get_device_name(index)}; "
            f"mean={result.mean().item():.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
