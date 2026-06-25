#!/usr/bin/env python3
"""Benchmark llama.cpp server using server-side timings (pp = prefill, tg = decode)."""
import os
import sys

try:
    import requests
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

BASE    = "http://qwen3.5-9b:8000"
API_KEY = os.environ.get("LLAMA_API_KEY", "changeme")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
MODEL   = "qwen3.5-9b"

PP_SIZES  = [128, 512, 2048, 8192]
TG_TOKENS = 128
REPS      = 3


def complete(prompt: str, max_tokens: int) -> dict:
    r = requests.post(
        f"{BASE}/v1/completions",
        headers=HEADERS,
        json={"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "stream": False},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def mean_tps(results: list, key: str) -> float:
    vals = [r["timings"][key] for r in results if "timings" in r]
    return sum(vals) / len(vals) if vals else 0.0


print("Warming up...", flush=True)
complete("Hello", 1)

print(f"\n{'test':<6} {'tokens':>8}  {'t/s':>10}")
print("-" * 28)

for pp in PP_SIZES:
    prompt = "word " * pp
    results = [complete(prompt, 1) for _ in range(REPS)]
    tps = mean_tps(results, "prompt_per_second")
    print(f"{'pp':<6} {pp:>8}  {tps:>10.1f}")

results = [complete("Summarize briefly:", TG_TOKENS) for _ in range(REPS)]
tps = mean_tps(results, "predicted_per_second")
print(f"{'tg':<6} {TG_TOKENS:>8}  {tps:>10.1f}")
