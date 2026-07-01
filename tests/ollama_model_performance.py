#!/usr/bin/env python3
"""Concurrent latency test for a local Ollama model.

This measures the latency of direct Ollama model generation calls, not the full
LangGraph workflow. Use it when you want to understand local model performance
with configurable concurrent users.

Usage:
    python tests/ollama_model_performance.py --model llama3.2:latest
    python tests/ollama_model_performance.py --requests 20 --concurrency 5
    python3 -B tests/ollama_model_performance.py --model llama3.2:latest --requests 10 --concurrency 10 --options '{"num_predict": 64, "temperature": 0}
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


def percentile(values: list[float], percentile_rank: float) -> float:
    """Return the nearest-rank percentile from a list of values."""
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = math.ceil((percentile_rank / 100) * len(sorted_values)) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {
            "avg_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }

    return {
        "avg_ms": round(statistics.mean(latencies_ms), 2),
        "p50_ms": round(percentile(latencies_ms, 50), 2),
        "p95_ms": round(percentile(latencies_ms, 95), 2),
        "min_ms": round(min(latencies_ms), 2),
        "max_ms": round(max(latencies_ms), 2),
    }


def call_ollama(
    base_url: str,
    model: str,
    prompt: str,
    timeout_seconds: float,
    options: dict[str, Any],
) -> dict[str, Any]:
    """Call Ollama /api/generate once and measure request latency."""
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }

    started_at = time.perf_counter()
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {error_body}") from exc

    elapsed_ms = (time.perf_counter() - started_at) * 1000
    return {
        "elapsed_ms": elapsed_ms,
        "response_chars": len(body.get("response", "")),
        "eval_count": body.get("eval_count"),
        "eval_duration_ns": body.get("eval_duration"),
        "prompt_eval_count": body.get("prompt_eval_count"),
        "prompt_eval_duration_ns": body.get("prompt_eval_duration"),
    }


def run_performance_test(
    requests_count: int,
    concurrency: int,
    warmup: int,
    base_url: str,
    model: str,
    prompt: str,
    timeout_seconds: float,
    options: dict[str, Any],
) -> dict[str, Any]:
    latencies_ms: list[float] = []
    errors: list[dict[str, str]] = []
    response_chars: list[int] = []
    eval_counts: list[int] = []

    print(
        f"Starting Ollama model performance test: "
        f"{requests_count} request(s), concurrency={concurrency}, warmup={warmup}"
    )

    def invoke(run_number: int, phase: str, start_event: threading.Event | None):
        if start_event:
            start_event.wait()

        try:
            result = call_ollama(
                base_url=base_url,
                model=model,
                prompt=prompt,
                timeout_seconds=timeout_seconds,
                options=options,
            )
            return {
                "ok": True,
                "run": str(run_number),
                "phase": phase,
                **result,
            }
        except Exception as exc:
            return {
                "ok": False,
                "run": str(run_number),
                "phase": phase,
                "elapsed_ms": 0.0,
                "error": repr(exc),
            }

    for warmup_index in range(warmup):
        result = invoke(warmup_index + 1, "warmup", None)
        if not result["ok"]:
            errors.append(
                {
                    "run": result["run"],
                    "phase": result["phase"],
                    "error": result["error"],
                }
            )
        print(
            f"{warmup_index + 1:>3}/{warmup} "
            f"warmup   {result['elapsed_ms']:>10.2f} ms"
        )

    measured_start_event = threading.Event()
    measured_started_at = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(invoke, request_index + 1, "measured", measured_start_event)
            for request_index in range(requests_count)
        ]

        measured_started_at = time.perf_counter()
        measured_start_event.set()

        for completed_count, future in enumerate(
            concurrent.futures.as_completed(futures),
            start=1,
        ):
            result = future.result()
            status = "ok" if result["ok"] else "failed"

            if result["ok"]:
                latencies_ms.append(result["elapsed_ms"])
                response_chars.append(result["response_chars"])
                if result.get("eval_count") is not None:
                    eval_counts.append(result["eval_count"])
            else:
                errors.append(
                    {
                        "run": result["run"],
                        "phase": result["phase"],
                        "error": result["error"],
                    }
                )

            print(
                f"{completed_count:>3}/{requests_count} "
                f"measured {status:<6} {result['elapsed_ms']:>10.2f} ms"
            )

    measured_duration_seconds = time.perf_counter() - measured_started_at
    successful_requests = len(latencies_ms)
    throughput = (
        successful_requests / measured_duration_seconds
        if measured_duration_seconds > 0
        else 0.0
    )

    summary = summarize_latencies(latencies_ms)
    return {
        "timestamp": datetime.now().isoformat(),
        "target": "ollama.api.generate",
        "base_url": base_url,
        "model": model,
        "prompt": prompt,
        "requests": requests_count,
        "concurrency": concurrency,
        "actual_concurrent_users": min(requests_count, concurrency),
        "warmup": warmup,
        "successful_requests": successful_requests,
        "failed_requests": len(errors),
        "success_rate": round(successful_requests / requests_count, 4)
        if requests_count
        else 0.0,
        "measured_duration_seconds": round(measured_duration_seconds, 2),
        "throughput_requests_per_second": round(throughput, 4),
        "avg_response_chars": round(statistics.mean(response_chars), 2)
        if response_chars
        else 0.0,
        "avg_eval_tokens": round(statistics.mean(eval_counts), 2)
        if eval_counts
        else 0.0,
        **summary,
        "errors": errors[:10],
    }


def print_summary(results: dict[str, Any]) -> None:
    print("\n" + "=" * 64)
    print("OLLAMA MODEL PERFORMANCE RESULTS")
    print("=" * 64)
    print(f"Target              : {results['target']}")
    print(f"Model               : {results['model']}")
    print(f"Requests            : {results['requests']}")
    print(f"Concurrency         : {results['concurrency']}")
    print(f"Actual Users        : {results['actual_concurrent_users']}")
    print(f"Successful Requests : {results['successful_requests']}")
    print(f"Failed Requests     : {results['failed_requests']}")
    print(f"Success Rate        : {results['success_rate']:.2%}")
    print(f"Measured Duration   : {results['measured_duration_seconds']:.2f} s")
    print(
        f"Throughput          : "
        f"{results['throughput_requests_per_second']:.4f} req/s"
    )
    print(f"Average Latency     : {results['avg_ms']:.2f} ms")
    print(f"P50 Latency         : {results['p50_ms']:.2f} ms")
    print(f"P95 Latency         : {results['p95_ms']:.2f} ms")
    print(f"Min Latency         : {results['min_ms']:.2f} ms")
    print(f"Max Latency         : {results['max_ms']:.2f} ms")
    print(f"Avg Response Chars  : {results['avg_response_chars']:.2f}")
    print(f"Avg Eval Tokens     : {results['avg_eval_tokens']:.2f}")
    print("=" * 64)


def parse_options(raw_options: str) -> dict[str, Any]:
    if not raw_options:
        return {}

    parsed = json.loads(raw_options)
    if not isinstance(parsed, dict):
        raise ValueError("--options must be a JSON object")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure p50/p95 latency for a local Ollama model."
    )
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--model", default="llama3.2:latest")
    parser.add_argument(
        "--prompt",
        default="Explain what LangGraph is in one concise paragraph.",
    )
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--options",
        default='{"num_predict": 128}',
        help='Ollama options JSON. Example: \'{"num_predict": 64, "temperature": 0}\'',
    )
    parser.add_argument("--output", default="ollama_performance_results.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.requests < 1:
        raise ValueError("--requests must be at least 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative")

    results = run_performance_test(
        requests_count=args.requests,
        concurrency=args.concurrency,
        warmup=args.warmup,
        base_url=args.base_url,
        model=args.model,
        prompt=args.prompt,
        timeout_seconds=args.timeout,
        options=parse_options(args.options),
    )
    print_summary(results)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results saved to {output_path}")

    return 0 if results["failed_requests"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
