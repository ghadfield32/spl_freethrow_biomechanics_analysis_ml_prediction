#!/usr/bin/env python3
"""
test_gpu.py

Check and benchmark a simple GPU operation in PyTorch and JAX.
"""

import time
import sys

def _pretty_capability(name):
    import torch
    major, minor = torch.cuda.get_device_capability()
    return f"{name} (sm_{major}{minor:02d})"

def test_pytorch(force_cpu: bool = False):
    import torch
    import time

    # — Detect device & arch —
    cuda_ok = torch.cuda.is_available() and not force_cpu
    arch_supported = False
    if cuda_ok:
        arch_supported = torch.cuda.get_device_capability()[0] <= 9  # sm_90 max in stable wheels

    if not (cuda_ok and arch_supported):
        reason = ("no CUDA" if not cuda_ok else "unsupported arch sm_120")
        print(f"[PyTorch] Falling back to CPU – {reason}")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        print(f"[PyTorch] Using {_pretty_capability(torch.cuda.get_device_name(0))}")

    # — Benchmark —
    size = (1000, 1000)
    a, b = (torch.randn(size, device=device) for _ in range(2))
    _ = a @ b  # warm-up
    start = time.time()
    (a @ b).sum().item()  # ensure reduction
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[PyTorch] matmul on {device} took {(time.time()-start)*1000:.2f} ms")



if __name__ == "__main__":
    print("=== GPU Test Script ===\n")
    test_pytorch()



