"""Per-request prefill-redundancy instrumentation for turboquant-serve.

Diagnostic only. Wraps ``LRUPromptCache.fetch_nearest_cache`` — mlx_lm's
prompt-cache reuse path — to measure, for every request, how much of the
prompt is actually re-prefilled vs. served from a cached prefix, and (the
number we care about) how many of those freshly-prefilled tokens were
*byte-identical to the previous turn's prefix* and could be eliminated by a
working prefix cache.

This is the measurement that decides whether "extend-without-trim prefix
reuse" is worth building for agentic clients (Claude Code / Aider), whose
prompts grow append-only every turn. Enable with ``--prefill-stats``.

``fetch_nearest_cache(model, tokens) -> (cache, rest)`` returns ``rest``, the
suffix the server must prefill, so ``len(tokens) - len(rest)`` is exactly what
the server reused and ``len(rest)`` is the fresh prefill. We additionally
compare ``tokens`` against the previous request's tokens to find the longest
common prefix (LCP): the gap ``LCP - reused`` is the redundant re-prefill a
prefix cache would recover.

Hardware-invariant: the redundancy *fraction* depends only on the client and
the conversation, not on RAM — so it can be measured on any machine (e.g. a
64 GB Mac) and the result transfers to the 16 GB mini. Only the absolute
prefill *seconds* differ between machines.
"""

from __future__ import annotations

import atexit
import json
import sys
import time
from typing import Any, List, Optional, TextIO


def _longest_common_prefix(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class PrefillStats:
    """Cumulative prefill-redundancy counters across a serving session."""

    def __init__(self, stats_file: Optional[str] = None,
                 log: TextIO = sys.stderr):
        self.stats_file = stats_file
        self.log = log
        self.n_requests = 0
        self.total_prompt = 0
        self.total_reused = 0       # tokens served from cache (server-actual)
        self.total_fresh = 0        # tokens freshly prefilled (== len(rest))
        self.total_recoverable = 0  # fresh tokens identical to prev-turn prefix
        self._prev_tokens: List[int] = []
        self._t0 = time.time()
        if stats_file:
            # Start the JSONL log fresh for this session.
            with open(stats_file, "w"):
                pass

    def record(self, tokens: List[int], rest: List[int]) -> dict:
        """Update counters from one ``fetch_nearest_cache`` call.

        ``rest`` is the suffix the server must prefill. Returns the per-request
        record (also emitted to the log / JSONL file).
        """
        total = len(tokens)
        fresh = len(rest)
        reused = total - fresh
        lcp_prev = _longest_common_prefix(tokens, self._prev_tokens)
        # Tokens re-prefilled this turn that were byte-identical to last turn's
        # prefix — exactly what a working prefix cache would have eliminated.
        recoverable = max(0, lcp_prev - reused)

        self.n_requests += 1
        self.total_prompt += total
        self.total_reused += reused
        self.total_fresh += fresh
        self.total_recoverable += recoverable
        self._prev_tokens = list(tokens)

        def _pct(x: int) -> float:
            return round(100 * x / total, 1) if total else 0.0

        rec = dict(
            req=self.n_requests,
            prompt_tokens=total,
            reused=reused,
            fresh_prefill=fresh,
            reuse_pct=_pct(reused),
            prev_lcp=lcp_prev,
            prev_lcp_pct=_pct(lcp_prev),
            recoverable=recoverable,
            recoverable_pct=_pct(recoverable),
        )
        self._emit(rec)
        return rec

    def _emit(self, rec: dict) -> None:
        self.log.write(
            f"[prefill-stats] req#{rec['req']} "
            f"prompt={rec['prompt_tokens']} tok | "
            f"reused={rec['reused']} ({rec['reuse_pct']}%) "
            f"fresh={rec['fresh_prefill']} | "
            f"prev-LCP={rec['prev_lcp']} ({rec['prev_lcp_pct']}%) | "
            f"recoverable={rec['recoverable']} ({rec['recoverable_pct']}% of prompt)\n"
        )
        self.log.flush()
        if self.stats_file:
            with open(self.stats_file, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def summary(self) -> str:
        if self.n_requests == 0:
            return "[prefill-stats] no requests recorded.\n"

        def _pct(x: int) -> float:
            return (100 * x / self.total_prompt) if self.total_prompt else 0.0

        fresh_after = self.total_fresh - self.total_recoverable
        drop = (100 * self.total_recoverable / self.total_fresh
                ) if self.total_fresh else 0.0
        return (
            f"\n[prefill-stats] ===== SESSION SUMMARY "
            f"({self.n_requests} requests, {time.time() - self._t0:.0f}s) =====\n"
            f"  total prompt tokens processed : {self.total_prompt:,}\n"
            f"  server reused (prefix cache)  : {self.total_reused:,} "
            f"({_pct(self.total_reused):.1f}%)\n"
            f"  server fresh-prefilled        : {self.total_fresh:,} "
            f"({_pct(self.total_fresh):.1f}%)\n"
            f"  recoverable via prefix reuse  : {self.total_recoverable:,} "
            f"({_pct(self.total_recoverable):.1f}% of all prompt tokens)\n"
            f"    -> a working prefix cache would cut fresh prefill "
            f"{self.total_fresh:,} -> {fresh_after:,} (-{drop:.0f}%)\n"
        )


def install(stats_file: Optional[str] = None,
            log: TextIO = sys.stderr) -> PrefillStats:
    """Monkeypatch ``LRUPromptCache.fetch_nearest_cache`` to record redundancy.

    Idempotent-safe: instrumentation failures never propagate into serving.
    Registers an ``atexit`` hook that prints the cumulative session summary.
    """
    from mlx_lm.models.cache import LRUPromptCache

    stats = PrefillStats(stats_file=stats_file, log=log)
    _orig = LRUPromptCache.fetch_nearest_cache

    def _wrapped(self, model: Any, tokens: List[int]):
        cache, rest = _orig(self, model, tokens)
        try:
            stats.record(list(tokens), rest)
        except Exception as e:  # never let instrumentation break serving
            log.write(f"[prefill-stats] (record skipped: {e})\n")
        return cache, rest

    LRUPromptCache.fetch_nearest_cache = _wrapped
    atexit.register(lambda: log.write(stats.summary()))
    return stats
