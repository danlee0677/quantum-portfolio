"""Migrate the author's saved HOPO-with-QAOA results into the current pipeline
format so ``profile.py --hopo-source json`` can ingest them like any
``results/portfolio_optimization_exp_*.json`` file.

The author's results live in ``results_old/cmaes_hubo_results/lambda_<L>/`` as one
``portfolio_optimization_results_batch_<n>.json`` per experiment, each a dict
keyed by the 0-based string experiment id. The per-experiment schema is nearly
identical to what ``experiments.py --method hopo`` writes today; the only deltas
(verified against experiments.py:171-303) are:

  * ``hyperparams`` lacks ``optimizer`` and ``lambda_budget``        -> add them
  * ``continuous_variables_solution_unconstrained`` is absent        -> set null

``qaoa_solution.result_with_budget`` entries lack the per-portfolio
``objective_value`` the current pipeline adds, but that field is additive and
unused by the loader, so it is left as-is.

All batch files are merged into a single current-convention file dropped into
``results/``:

    portfolio_optimization_exp_CMAES_<lambda>_<start>_<end>_<stamp>.json

``<stamp>`` is a required CLI argument (scripts here cannot read the clock and a
``YYYYMMDD_HHMMSS``-shaped token keeps profile.py's newest-first sort
well-defined). ``lambda_1`` is the canonical budget-penalty baseline.

Usage:
    uv run python migrate_author_hopo_results.py <stamp> [--lambda-folder lambda_1]
    # e.g.
    uv run python migrate_author_hopo_results.py 20260610_120000
"""

import argparse
import glob
import json
import os
import re


SOURCE_ROOT = os.path.join("results_old", "cmaes_hubo_results")
RESULTS_DIR = "results"
OPTIMIZER = "CMAES"

# Map a lambda_<L> folder name to the float lambda_budget it represents.
# The folders are encoded without the decimal point (lambda_0001 == 0.001).
_LAMBDA_FROM_FOLDER = {
    "lambda_0001": 0.001,
    "lambda_001": 0.01,
    "lambda_01": 0.1,
    "lambda_09": 0.9,
    "lambda_1": 1.0,
    "lambda_10": 10.0,
    "lambda_100": 100.0,
    "lambda_1000": 1000.0,
}


def lambda_for_folder(folder):
    if folder in _LAMBDA_FROM_FOLDER:
        return _LAMBDA_FROM_FOLDER[folder]
    raise ValueError(
        f"unknown lambda folder {folder!r}; expected one of {sorted(_LAMBDA_FROM_FOLDER)}"
    )


def migrate_entry(entry, lambda_budget):
    """Bring one author per-experiment entry up to the current pipeline shape.

    Mutates a shallow copy so the input dict is left untouched. Entries that are
    error stubs (``{"error": ...}``) are passed through unchanged.
    """
    if "hyperparams" not in entry:
        return dict(entry)  # e.g. an insufficient_valid_tickers stub
    migrated = dict(entry)
    hp = dict(migrated["hyperparams"])
    hp.setdefault("optimizer", OPTIMIZER)
    hp.setdefault("lambda_budget", lambda_budget)
    migrated["hyperparams"] = hp
    # Current pipeline always writes this key (null when not computed).
    migrated.setdefault("continuous_variables_solution_unconstrained", None)
    return migrated


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stamp",
                        help="YYYYMMDD_HHMMSS-shaped token embedded in the output "
                             "filename (used by profile.py's newest-first sort).")
    parser.add_argument("--lambda-folder", default="lambda_1",
                        help="Author lambda_<L> subfolder to migrate (default lambda_1, "
                             "the canonical lambda_budget=1.0 baseline).")
    parser.add_argument("--source-root", default=SOURCE_ROOT,
                        help=f"Root holding the lambda_<L> folders (default {SOURCE_ROOT}).")
    parser.add_argument("--results-dir", default=RESULTS_DIR,
                        help=f"Output directory (default {RESULTS_DIR}).")
    args = parser.parse_args()

    if not re.fullmatch(r"\d{8}_\d{6}", args.stamp):
        parser.error("stamp must be YYYYMMDD_HHMMSS-shaped, e.g. 20260610_120000")

    lambda_budget = lambda_for_folder(args.lambda_folder)
    src_dir = os.path.join(args.source_root, args.lambda_folder)
    pattern = os.path.join(src_dir, "portfolio_optimization_results_batch_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        parser.error(f"no author batch files found at {pattern}")

    merged = {}
    for path in files:
        with open(path, "r") as f:
            data = json.load(f)
        for key, entry in data.items():
            if key in merged:
                print(f"  WARNING: duplicate id {key} (from {os.path.basename(path)}); "
                      f"keeping the earlier one")
                continue
            merged[key] = migrate_entry(entry, lambda_budget)

    ids = sorted(int(k) for k in merged)
    start, end = ids[0], ids[-1]
    os.makedirs(args.results_dir, exist_ok=True)
    out_name = (
        f"portfolio_optimization_exp_{OPTIMIZER}_{str(lambda_budget)}_"
        f"{start}_{end}_{args.stamp}.json"
    )
    out_path = os.path.join(args.results_dir, out_name)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=4)

    print(f"Migrated {len(merged)} experiments from {args.lambda_folder} "
          f"(lambda_budget={lambda_budget}); ids {start}..{end}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
