#!/usr/bin/env python3
"""Performance runner for the LangGraph financial agent workflow.

This measures end-to-end latency for FinancialAgentWorkflow.analyze(), which
includes the compiled LangGraph app plus the project nodes it calls. It prints
p50/p95 latency and writes a JSON report for later comparison.

Usage:
    python tests/performance_test1.py --requests 10 --ticker AAPL
    python tests/performance_test1.py --requests 20 --warmup 2 --output perf.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def percentile(values: list[float], percentile_rank: float) -> float:
    """Return the nearest-rank percentile from a list of latency values."""
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = round((percentile_rank / 100) * (len(sorted_values) - 1))
    return sorted_values[index]


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    """Build common latency metrics in milliseconds."""
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


def run_performance_test(
    num_requests: int,
    warmup_requests: int,
    concurrency: int,
    ticker: str,
    query: str,
) -> dict[str, Any]:
    """Run the workflow repeatedly and return performance metrics."""
    from src.workflow import FinancialAgentWorkflow

    latencies_ms: list[float] = []
    errors: list[dict[str, str]] = []

    print(
        f"Starting LangGraph performance test: "
        f"{num_requests} measured request(s), {warmup_requests} warmup request(s), "
        f"concurrency={concurrency}"
    )

    def invoke_workflow(run_number: int, phase: str) -> dict[str, Any]:
        workflow = FinancialAgentWorkflow()
        started_at = time.perf_counter()
        try:
            result = workflow.analyze(ticker=ticker, query=query)
            elapsed_ms = (time.perf_counter() - started_at) * 1000

            if result.get("error"):
                return {
                    "ok": False,
                    "run": str(run_number),
                    "phase": phase,
                    "elapsed_ms": elapsed_ms,
                    "error": str(result["error"]),
                }

            return {
                "ok": True,
                "run": str(run_number),
                "phase": phase,
                "elapsed_ms": elapsed_ms,
            }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            return {
                "ok": False,
                "run": str(run_number),
                "phase": phase,
                "elapsed_ms": elapsed_ms,
                "error": repr(exc),
            }

    for warmup_index in range(warmup_requests):
        run_result = invoke_workflow(warmup_index + 1, "warmup")
        if not run_result["ok"]:
            errors.append(
                {
                    "run": run_result["run"],
                    "phase": run_result["phase"],
                    "error": run_result["error"],
                }
            )

        print(
            f"{warmup_index + 1:>3}/{warmup_requests} "
            f"warmup   {run_result['elapsed_ms']:>10.2f} ms"
        )

    measured_started_at = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(invoke_workflow, request_index + 1, "measured")
            for request_index in range(num_requests)
        ]

        for completed_count, future in enumerate(
            concurrent.futures.as_completed(futures),
            start=1,
        ):
            run_result = future.result()

            if run_result["ok"]:
                latencies_ms.append(run_result["elapsed_ms"])
            else:
                errors.append(
                    {
                        "run": run_result["run"],
                        "phase": run_result["phase"],
                        "error": run_result["error"],
                    }
                )

            status = "ok" if run_result["ok"] else "failed"
            print(
                f"{completed_count:>3}/{num_requests} "
                f"measured {status:<6} {run_result['elapsed_ms']:>10.2f} ms"
            )

    measured_elapsed_seconds = time.perf_counter() - measured_started_at
    throughput = (
        len(latencies_ms) / measured_elapsed_seconds
        if measured_elapsed_seconds > 0
        else 0.0
    )

    summary = summarize_latencies(latencies_ms)
    return {
        "timestamp": datetime.now().isoformat(),
        "target": "FinancialAgentWorkflow.analyze",
        "ticker": ticker.upper(),
        "query": query,
        "concurrency": concurrency,
        "warmup_requests": warmup_requests,
        "measured_requests": num_requests,
        "successful_requests": len(latencies_ms),
        "failed_requests": len(errors),
        "measured_duration_seconds": round(measured_elapsed_seconds, 2),
        "throughput_requests_per_second": round(throughput, 4),
        "success_rate": round(len(latencies_ms) / num_requests, 4)
        if num_requests
        else 0.0,
        **summary,
        "errors": errors[:10],
    }


def print_summary(results: dict[str, Any]) -> None:
    """Print a compact terminal summary."""
    print("\n" + "=" * 64)
    print("LANGGRAPH PERFORMANCE RESULTS")
    print("=" * 64)
    print(f"Target              : {results['target']}")
    print(f"Ticker              : {results['ticker']}")
    print(f"Concurrency         : {results['concurrency']}")
    print(f"Measured Requests   : {results['measured_requests']}")
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
    print("=" * 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure p50/p95 latency for the LangGraph financial workflow."
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=10,
        help="Number of measured workflow runs. Default: 10.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs excluded from metrics. Default: 1.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of measured workflow runs to execute at once. Default: 1.",
    )
    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Ticker symbol passed into the workflow. Default: AAPL.",
    )
    parser.add_argument(
        "--query",
        default="Should I invest in this stock?",
        help="Analysis query passed into the workflow.",
    )
    parser.add_argument(
        "--output",
        default="performance_results.json",
        help="JSON output file path. Default: performance_results.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.requests < 1:
        raise ValueError("--requests must be at least 1")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")

    results = run_performance_test(
        num_requests=args.requests,
        warmup_requests=args.warmup,
        concurrency=args.concurrency,
        ticker=args.ticker,
        query=args.query,
    )
    print_summary(results)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results saved to {output_path}")

    return 0 if results["failed_requests"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
