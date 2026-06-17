"""Generate a ring-XY-only experiment dataset (qubit count >= 7).

The ring-XY cardinality-selection QAOA (``RingXYCardinalityQAOA``) uses exactly
**one qubit per stock**, so its qubit count equals ``N = len(stocks)`` — unlike the
original HOPO method, whose qubit count is inflated by per-asset integer log-encoding.
The shipped ``experiments_data.json`` was bucketed on the HOPO qubit count and so only
contains 2-6 stock problems (N <= 6), i.e. nothing reaches 7 qubits for ring-XY.

This script produces a separate dataset whose problems have N >= 7 stocks, bucketed
directly on N (= ring-XY qubit count). The per-entry JSON schema is identical to the
original so all downstream tooling (``experiments.py --method ring_xy``, ``profile.py``)
works unchanged:

    {"max_qubits": N, "budget": int, "stocks": [N tickers],
     "start": "2015-01-01", "end": "2025-01-01", "n_stocks": N}

``max_qubits`` is set to ``N`` so the qubit-count field reflects the ring-XY qubit
count (and any bucket-by-max_qubits tooling sees the correct value).

Run with uv (the project is uv-managed):

    uv run python generate_ring_xy_experiments.py
    uv run python generate_ring_xy_experiments.py --n-min 7 --n-max 15 --per-n 10
"""

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np

from data_provider import fetch_close_prices

# The Dow-30 universe, exactly as printed by
# ``PyTickerSymbols().get_dow_jones_nyc_yahoo_tickers()`` in
# generate_portfolio_experiments.ipynb (the same universe that produced the original
# dataset). Hardcoded so this script needs no extra dependency; if pytickersymbols is
# installed we use its (possibly more current) list instead.
DOW_TICKERS = [
    "MMM", "AXP", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "GS", "HD",
    "IBM", "INTC", "JNJ", "JPM", "MCD", "NKE", "MRK", "MSFT", "PG", "TRV",
    "UNH", "VZ", "V", "WMT", "WBA", "DIS", "AMGN", "HON", "CRM", "DOW",
]


def dow_universe():
    """Return the Dow-30 tickers, preferring pytickersymbols if available."""
    try:
        from pytickersymbols import PyTickerSymbols
        return list(PyTickerSymbols().get_dow_jones_nyc_yahoo_tickers())
    except ImportError:
        return list(DOW_TICKERS)


def build_clean_pool(start, end):
    """Return (clean_pool, prices_now) for the Dow-30 universe.

    Downloads all Dow tickers once with the robust, retry-aware fetcher and keeps
    only those that survive (drops genuinely-dead tickers such as WBA, exactly as
    ``experiments.py`` does at run time). This guarantees that a sampled N-stock
    problem still has N live tickers when the experiment is later run.
    """
    universe = dow_universe()
    print(f"Dow universe: {len(universe)} tickers; fetching {start}..{end} ...")
    fetched = fetch_close_prices(universe, start=start, end=end,
                                 auto_adjust=False, progress=False)
    clean_pool = list(fetched.close.columns)
    if fetched.dropped:
        print(f"Dropped (dead/unavailable): {fetched.dropped} "
              f"(delisted={fetched.delisted}, unavailable={fetched.unavailable})")
    prices_now = fetched.close.iloc[-1]  # Series indexed by ticker
    print(f"Clean pool: {len(clean_pool)} tickers -> {clean_pool}")
    return clean_pool, prices_now


def load_existing(out_path):
    """Load an existing output file into per-N buckets (for idempotent top-up)."""
    buckets = defaultdict(list)
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            for entry in json.load(f).get("data", []):
                buckets[entry["n_stocks"]].append(entry)
        total = sum(len(v) for v in buckets.values())
        print(f"Loaded {total} existing problems from {out_path}")
    return buckets


def generate(clean_pool, prices_now, *, n_min, n_max, per_n, budget_cap,
             start, end, buckets):
    """Top up each N-bucket (n_min..n_max) to ``per_n`` problems."""
    if len(clean_pool) < n_max:
        sys.exit(f"Clean pool has only {len(clean_pool)} tickers; need >= n_max={n_max} "
                 f"to sample distinct stocks for the largest problems.")

    for N in range(n_min, n_max + 1):
        while len(buckets[N]) < per_n:
            stocks = random.sample(clean_pool, N)
            lo = int(math.ceil(float(max(prices_now[s] for s in stocks))))
            # Guard: keep generation from wedging if the priciest single share is
            # somehow >= the cap (it never is for the Dow-30, but stay safe).
            cap = budget_cap if lo < budget_cap else lo + 1000
            budget = random.randint(lo, cap)
            buckets[N].append({
                "max_qubits": N,
                "budget": budget,
                "stocks": stocks,
                "start": start,
                "end": end,
                "n_stocks": N,
            })
    return buckets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="experiments_data_ring_xy.json",
                        help="Output dataset path.")
    parser.add_argument("--seed", type=int, default=2,
                        help="RNG seed (default 2; the original dataset used seed 1).")
    parser.add_argument("--n-min", type=int, default=7,
                        help="Smallest N (= ring-XY qubit count). Default 7.")
    parser.add_argument("--n-max", type=int, default=15,
                        help="Largest N (= ring-XY qubit count). Default 15.")
    parser.add_argument("--per-n", type=int, default=10,
                        help="Problems per N bucket. Default 10.")
    parser.add_argument("--budget-cap", type=int, default=6000,
                        help="Upper bound on the random budget. Default 6000.")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-01-01")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    clean_pool, prices_now = build_clean_pool(args.start, args.end)
    buckets = load_existing(args.out)
    buckets = generate(clean_pool, prices_now,
                       n_min=args.n_min, n_max=args.n_max, per_n=args.per_n,
                       budget_cap=args.budget_cap, start=args.start, end=args.end,
                       buckets=buckets)

    data = []
    for N in sorted(buckets):
        data.extend(buckets[N])
    with open(args.out, "w") as f:
        json.dump({"data": data}, f, indent=2)

    print(f"\nWrote {len(data)} problems to {args.out}")
    for N in sorted(buckets):
        print(f"  N={N} (ring-XY qubits={N}): {len(buckets[N])} problems")


if __name__ == "__main__":
    main()
