#!/usr/bin/env python3
"""Benchmark NVIDIA NIM free chat endpoints from the current catalog.

The final CLI table is intentionally small:
    model | score | uptime | intelligence | avg time | ttft | best time | throughput

Definitions:
- score: local relative performance score (0-100) for THIS benchmark run.
- uptime: successful streaming requests / attempted requests.
- intelligence: Artificial Analysis Intelligence Index snapshot (external, not measured here).
- avg time: mean end-to-end stream duration across successful runs.
- ttft: mean time to first generated token/delta, including reasoning tokens when streamed.
- best time: fastest successful end-to-end stream duration.
- throughput: mean completion tokens/second when NVIDIA returns completion token usage.

Scope:
This script benchmarks general/agentic text-generating Free Endpoints that use
OpenAI-compatible /v1/chat/completions. The NVIDIA catalog also contains free
embedding, reranking, speech, CV, biology, safety, and other specialized APIs;
those are deliberately excluded because their latency is not comparable with a
chat-completion benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE = "https://integrate.api.nvidia.com/v1"
CHAT_URL = BASE + "/chat/completions"
MODELS_URL = BASE + "/models"

# Free chat/text-generation endpoints visible in the NVIDIA catalog on 2026-10-23.
# Domain-specific/safety-only endpoints are excluded so the same text prompt is meaningful.
DEFAULT_MODELS = [
    "minimaxai/minimax-m3.1-flash",
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "stepfun-ai/step-5-preview",
    "deepseek-ai/deepseek-v4.1-flash",
    "z-ai/glm-5.3",
    "z-ai/glm-5.3-flash",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "meta/muse-glimmer-30b",
    "poolside/laguna-xs-2.1",
    "google/diffusiongemma-26b-a4b-it",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "google/gemma-4-31b-it",
    "openai/gpt-oss-20b",
]

# Artificial Analysis Intelligence Index v4.3.2 snapshot, checked 2026-10-23.
# None means no sufficiently grounded matching score was found for this exact model.
# estimated=True is rendered with an asterisk.
MODEL_META: dict[str, dict[str, Any]] = {
    "minimaxai/minimax-m3.1-flash": {"intelligence": 43, "estimated": False},
    "nvidia/nemotron-3.5-super-vl-120b-a12b": {"intelligence": 26, "estimated": False},
    "stepfun-ai/step-5-preview": {"intelligence": 44, "estimated": False},
    "deepseek-ai/deepseek-v4.1-flash": {"intelligence": 39, "estimated": False},
    "z-ai/glm-5.3": {"intelligence": 45, "estimated": False},
    "z-ai/glm-5.3-flash": {"intelligence": 42, "estimated": False},
    "nvidia/nemotron-3.5-lightning-30b-a3b": {"intelligence": 13, "estimated": False},
    "meta/muse-glimmer-30b": {"intelligence": 17, "estimated": False},
    "poolside/laguna-xs-2.1": {"intelligence": None, "estimated": False},
    "google/diffusiongemma-26b-a4b-it": {"intelligence": None, "estimated": False},
    "nvidia/nemotron-3-ultra-550b-a55b": {"intelligence": 23, "estimated": False},
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning": {"intelligence": 10, "estimated": True},
    "google/gemma-4-31b-it": {"intelligence": 15, "estimated": False},
    "openai/gpt-oss-20b": {"intelligence": 9, "estimated": False},
}

# Keep the prompt stable across models and runs. A bounded answer makes total-time
# and throughput comparisons less sensitive to model verbosity.
BENCH_PROMPT = (
    "You are a senior software engineer. In exactly 6 concise bullet points, explain "
    "how to make an in-memory TTL LRU cache thread-safe under concurrent reads and writes. "
    "Do not include code. Keep the complete answer between 100 and 140 words."
)

SCORE_WEIGHTS = {
    "uptime_pct": 0.35,       # higher is better
    "ttft_s": 0.30,           # lower is better
    "avg_time_s": 0.15,       # lower is better
    "throughput_tps": 0.20,   # higher is better
}


def mean_or_none(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def request_json(url: str, api_key: str, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "nim-latency-benchmark/3.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _stream_request(
    model: str,
    api_key: str,
    max_tokens: int,
    timeout: float,
    include_usage: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": BENCH_PROMPT}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.2,
        "top_p": 0.95,
    }
    if include_usage:
        payload["stream_options"] = {"include_usage": True}

    req = urllib.request.Request(
        CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "nim-latency-benchmark/3.0",
        },
    )

    started = time.perf_counter()
    first_generated_at: float | None = None
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    finish_reason: str | None = None
    generated_chars = 0

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            http_code = getattr(resp, "status", None)
            if http_code != 200:
                body = resp.read(1000).decode("utf-8", "replace")
                return {
                    "model": model,
                    "status": f"HTTP {http_code}: {body[:220]}",
                    "total_s": round(time.perf_counter() - started, 4),
                }

            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue

                usage = obj.get("usage") or {}
                if isinstance(usage.get("completion_tokens"), int):
                    completion_tokens = usage["completion_tokens"]
                if isinstance(usage.get("prompt_tokens"), int):
                    prompt_tokens = usage["prompt_tokens"]

                now = time.perf_counter()
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    reasoning = (
                        delta.get("reasoning_content")
                        or delta.get("reasoning")
                        or delta.get("thinking")
                    )
                    content = delta.get("content")

                    generated = False
                    if isinstance(reasoning, str) and reasoning:
                        generated = True
                        generated_chars += len(reasoning)
                    if isinstance(content, str) and content:
                        generated = True
                        generated_chars += len(content)

                    if generated and first_generated_at is None:
                        first_generated_at = now
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice.get("finish_reason"))

        ended = time.perf_counter()
        generation_s = ended - first_generated_at if first_generated_at is not None else None
        tokens_per_s = None
        if (
            isinstance(completion_tokens, int)
            and completion_tokens >= 0
            and generation_s is not None
            and generation_s > 0
        ):
            tokens_per_s = completion_tokens / generation_s

        if first_generated_at is None:
            return {
                "model": model,
                "status": "NO_GENERATED_TOKEN",
                "total_s": round(ended - started, 4),
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "finish_reason": finish_reason,
            }

        return {
            "model": model,
            "status": "OK" if include_usage else "OK_NO_USAGE",
            "total_s": round(ended - started, 4),
            "ttft_s": round(first_generated_at - started, 4),
            "generation_s": round(generation_s, 4) if generation_s is not None else None,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "tokens_per_s": round(tokens_per_s, 2) if tokens_per_s is not None else None,
            "generated_chars": generated_chars,
            "finish_reason": finish_reason,
        }

    except urllib.error.HTTPError as exc:
        body = exc.read(1200).decode("utf-8", "replace").replace("\n", " ")
        return {
            "model": model,
            "status": f"HTTP {exc.code}: {body[:280]}",
            "http_code": exc.code,
            "error_body": body[:1200],
            "total_s": round(time.perf_counter() - started, 4),
        }
    except Exception as exc:
        return {
            "model": model,
            "status": f"{type(exc).__name__}: {str(exc)[:240]}",
            "total_s": round(time.perf_counter() - started, 4),
        }


def stream_once(
    model: str,
    api_key: str,
    max_tokens: int,
    timeout: float,
    usage_capability: dict[str, bool],
) -> dict[str, Any]:
    include_usage = usage_capability.get(model, True)
    result = _stream_request(model, api_key, max_tokens, timeout, include_usage)
    if str(result.get("status", "")).startswith("OK"):
        return result

    body = str(result.get("error_body") or "").lower()
    if include_usage and result.get("http_code") in (400, 422) and any(
        marker in body
        for marker in ("stream_options", "extra_forbidden", "unexpected", "unknown field")
    ):
        usage_capability[model] = False
        retry = _stream_request(model, api_key, max_tokens, timeout, False)
        retry["retried_without_stream_usage"] = True
        return retry
    return result


def summarize_model(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    model_rows = [r for r in rows if r.get("model") == model]
    ok = [r for r in model_rows if str(r.get("status", "")).startswith("OK")]

    def values(key: str) -> list[float]:
        return [float(r[key]) for r in ok if isinstance(r.get(key), (int, float))]

    meta = MODEL_META.get(model, {})
    return {
        "model": model,
        "score": 0.0,
        "runs": len(model_rows),
        "successful_runs": len(ok),
        "uptime_pct": round(100.0 * len(ok) / max(1, len(model_rows)), 1),
        "intelligence": meta.get("intelligence"),
        "intelligence_estimated": bool(meta.get("estimated", False)),
        "avg_time_s": mean_or_none(values("total_s")),
        "ttft_s": mean_or_none(values("ttft_s")),
        "best_time_s": round(min(values("total_s")), 4) if values("total_s") else None,
        "throughput_tps": mean_or_none(values("tokens_per_s")),
        "last_error": next(
            (str(r.get("status")) for r in reversed(model_rows) if not str(r.get("status", "")).startswith("OK")),
            None,
        ),
    }


def _normalized(values: dict[str, float], higher_is_better: bool) -> dict[str, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi == lo:
        return {key: 1.0 for key in values}
    if higher_is_better:
        return {key: (value - lo) / (hi - lo) for key, value in values.items()}
    return {key: (hi - value) / (hi - lo) for key, value in values.items()}


def apply_relative_scores(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(r) for r in summaries]
    by_model = {r["model"]: r for r in rows}

    uptime = {
        r["model"]: float(r["uptime_pct"])
        for r in rows
        if isinstance(r.get("uptime_pct"), (int, float))
    }
    ttft = {
        r["model"]: float(r["ttft_s"])
        for r in rows
        if isinstance(r.get("ttft_s"), (int, float))
    }
    avg_time = {
        r["model"]: float(r["avg_time_s"])
        for r in rows
        if isinstance(r.get("avg_time_s"), (int, float))
    }
    throughput = {
        r["model"]: float(r["throughput_tps"])
        for r in rows
        if isinstance(r.get("throughput_tps"), (int, float))
    }

    normalized = {
        "uptime_pct": _normalized(uptime, True),
        "ttft_s": _normalized(ttft, False),
        "avg_time_s": _normalized(avg_time, False),
        "throughput_tps": _normalized(throughput, True),
    }

    for model, row in by_model.items():
        if row.get("uptime_pct", 0) <= 0:
            row["score"] = 0.0
            continue
        score = 0.0
        for metric, weight in SCORE_WEIGHTS.items():
            # Missing measurements earn zero for that component rather than being guessed.
            score += weight * normalized[metric].get(model, 0.0)
        row["score"] = round(score * 100.0, 1)

    return rows


def fmt_num(value: Any, digits: int = 2) -> str:
    if not isinstance(value, (int, float)):
        return "N/A"
    return f"{float(value):.{digits}f}"


def intelligence_text(row: dict[str, Any]) -> str:
    value = row.get("intelligence")
    if not isinstance(value, (int, float)):
        return "N/A"
    suffix = "*" if row.get("intelligence_estimated") else ""
    return f"{value:g}{suffix}"


def render_table(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda r: (-float(r.get("score") or 0), r["model"]))
    model_width = min(52, max(5, *(len(r["model"]) for r in ordered))) if ordered else 5

    header = (
        f"{'model':<{model_width}} | {'score':>5} | {'uptime':>7} | {'intelligence':>12} | "
        f"{'avg time':>8} | {'ttft':>7} | {'best time':>9} | {'throughput':>10}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in ordered:
        model = r["model"]
        if len(model) > model_width:
            model = model[: max(1, model_width - 1)] + "…"
        lines.append(
            f"{model:<{model_width}} | "
            f"{fmt_num(r.get('score'), 1):>5} | "
            f"{fmt_num(r.get('uptime_pct'), 1) + '%':>7} | "
            f"{intelligence_text(r):>12} | "
            f"{fmt_num(r.get('avg_time_s'), 2) + 's' if isinstance(r.get('avg_time_s'), (int, float)) else 'N/A':>8} | "
            f"{fmt_num(r.get('ttft_s'), 2) + 's' if isinstance(r.get('ttft_s'), (int, float)) else 'N/A':>7} | "
            f"{fmt_num(r.get('best_time_s'), 2) + 's' if isinstance(r.get('best_time_s'), (int, float)) else 'N/A':>9} | "
            f"{fmt_num(r.get('throughput_tps'), 1) + ' t/s' if isinstance(r.get('throughput_tps'), (int, float)) else 'N/A':>10}"
        )
    return "\n".join(lines)


def save_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_models(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_MODELS)
    models = [item.strip() for item in value.split(",") if item.strip()]
    if not models:
        raise ValueError("--models did not contain any model IDs")
    return list(dict.fromkeys(models))


def main() -> None:
    parser = argparse.ArgumentParser(description="NVIDIA NIM Free Endpoint latency benchmark")
    parser.add_argument("--runs", type=int, default=3, help="measured streaming requests per model")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--pause", type=float, default=0.40, help="pause between requests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="nim-benchmark")
    parser.add_argument(
        "--models",
        help="comma-separated NVIDIA model IDs; default is the curated Free Endpoint chat set",
    )
    parser.add_argument(
        "--force-unlisted",
        action="store_true",
        help="benchmark requested models even if GET /v1/models does not list them",
    )
    args = parser.parse_args()

    if args.runs < 1 or args.max_tokens < 1 or args.timeout <= 0 or args.pause < 0:
        parser.error("runs/max-tokens must be >= 1, timeout > 0, pause >= 0")

    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        print("ERROR: NVIDIA_API_KEY is not set.", file=sys.stderr)
        print("Run: export NVIDIA_API_KEY='nvapi-...'", file=sys.stderr)
        raise SystemExit(2)

    try:
        requested_models = parse_models(args.models)
    except ValueError as exc:
        parser.error(str(exc))

    available: set[str] | None = None
    try:
        data = request_json(MODELS_URL, api_key, min(args.timeout, 30.0))
        available = {m.get("id") for m in data.get("data", []) if isinstance(m.get("id"), str)}
    except Exception as exc:
        print(f"WARNING: GET /v1/models failed: {type(exc).__name__}: {exc}")

    if available is not None and not args.force_unlisted:
        models = [m for m in requested_models if m in available]
        skipped = [m for m in requested_models if m not in available]
    else:
        models = requested_models
        skipped = []

    if skipped:
        print("Skipping models not returned by GET /v1/models for this API key:")
        for model in skipped:
            print(f"  - {model}")
        print()

    if not models:
        print("ERROR: none of the requested models are available for this API key.", file=sys.stderr)
        raise SystemExit(3)

    print("NVIDIA NIM FREE ENDPOINT LATENCY BENCHMARK")
    print(f"Target : {CHAT_URL}")
    print(f"Models : {len(models)}")
    print(f"Runs   : {args.runs}/model, max_tokens={args.max_tokens}")
    print("TTFT   : first generated reasoning/content delta")
    print("Score  : 35% uptime + 30% TTFT + 15% avg time + 20% throughput (relative to this run)")
    print("Intel  : Artificial Analysis Intelligence Index snapshot 2026-10-23; * = estimated\n")

    rng = random.Random(args.seed)
    usage_capability: dict[str, bool] = {}
    raw_rows: list[dict[str, Any]] = []

    for run in range(1, args.runs + 1):
        order = list(models)
        rng.shuffle(order)
        for model in order:
            row = stream_once(model, api_key, args.max_tokens, args.timeout, usage_capability)
            row["run"] = run
            raw_rows.append(row)
            status = str(row.get("status", ""))
            if status.startswith("OK"):
                tps = fmt_num(row.get("tokens_per_s"), 1)
                print(
                    f"[{run}/{args.runs}] {model:<48} OK  "
                    f"total={fmt_num(row.get('total_s'), 2)}s  "
                    f"ttft={fmt_num(row.get('ttft_s'), 2)}s  tps={tps}"
                )
            else:
                print(f"[{run}/{args.runs}] {model:<48} ERROR  {status[:100]}")
            if args.pause:
                time.sleep(args.pause)

    summaries = [summarize_model(model, raw_rows) for model in models]
    summaries = apply_relative_scores(summaries)

    print("\n" + render_table(summaries))
    print("\nNotes:")
    print("  - Score is relative to the models in this exact run; do not compare scores across separate runs.")
    print("  - Intelligence is external quality data and is intentionally NOT part of the latency score.")
    print("  - Throughput is N/A if a model endpoint does not return completion_tokens in streaming usage.")
    print("  - GLM-5.3 is shown because it is in the uploaded/current catalog, but NVIDIA currently marks it for deprecation.")

    prefix = Path(args.output)
    summary_path = str(prefix) + "-summary.csv"
    raw_path = str(prefix) + "-runs.csv"
    json_path = str(prefix) + ".json"
    save_csv(summary_path, summaries)
    save_csv(raw_path, raw_rows)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "target": CHAT_URL,
                "config": {
                    "runs": args.runs,
                    "max_tokens": args.max_tokens,
                    "timeout_s": args.timeout,
                    "pause_s": args.pause,
                    "models": models,
                    "score_weights": SCORE_WEIGHTS,
                },
                "raw_runs": raw_rows,
                "summary": summaries,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {raw_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
