"""Rank-correlation audit: ring-XY selection energy vs post-allocation objective.

For each problem instance and each cardinality K, this measures how well the
ring-XY cardinality QAOA's *selection energy* ranks asset subsets against the
*post-allocation portfolio objective*, by computing the Kendall-tau and
Spearman-rho rank correlation between the two quantities over ALL C(N,K) subsets
of size K. It does this under three selection weightings -- "budget" (the current
default), "equal_split", and "unit" -- so they can be compared head to head.

This is a NEW, STANDALONE, strictly READ-ONLY script: it imports
``RingXYCardinalityQAOA`` and reuses its methods unchanged, reads
``experiments_data.json``, and writes only its own CSV/PNG artifacts under
``results/``. No existing code or result file is modified.

Sign convention (verified at ring_xy.py:117). ``get_objective_value`` builds the
standard minimization cost ``-mu.w + (ra/2) w'Sigma w - (ra/6) S + (ra/24) Kurt``
and returns its *negation*, so a HIGHER ``post_objective`` is a BETTER portfolio
("Maximized utility"). The selection energy is the opposite orientation: the QAOA
*minimizes* it (``solve_selection_exactly`` takes argmin on energy, argmax on
post_objective). Therefore a GOOD selection ranking -- low energy paired with high
post_objective -- yields a NEGATIVE Kendall-tau / Spearman-rho. We store the raw
``kendall_tau`` as the primary metric and also emit ``tau_oriented = -kendall_tau``
so that +1 reads as "perfect".

Data-source choice: RECOMPUTE. The saved results/ring_xy_exp_*.json store only the
per-K optima/bounds (not the full per-subset pairs) and omit the coskewness/
cokurtosis tensors, so the correlations cannot be reconstructed from them; they are
used only by --verify as a cross-check reference. Recompute is cheap (N<=10 ->
<=2**N allocator calls per instance, no QAOA).

Usage (uv-managed project, run via uv):
    uv run python rank_correlation_audit.py <start> <end>     # 0-based inclusive
    uv run python rank_correlation_audit.py                   # all problems
    uv run python rank_correlation_audit.py 0 2 --no-figure   # quick smoke test
    uv run python rank_correlation_audit.py 0 9 --verify      # self-test checks
"""

import argparse
import csv
import json
import math
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.stats import kendalltau, spearmanr

from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov

from coskweness_cokurtosis import cokurtosis, coskewness
from data_provider import fetch_close_prices
from ring_xy import RingXYCardinalityQAOA
from utils import hamming_weight_indices

RISK_AVERSION = 0.1          # matches the hardcoded value in experiments.py
ENUM_MAX_N = 16              # guard: worst case is 2**N allocator calls / instance
WEIGHTINGS = ["budget", "equal_split", "unit"]
# Ids whose unit-weighting K=1 inversion is documented in ring_xy.py
# (_approximation_ratios) / experiments.py: id 1 = TRV/NKE.
INVERSION_IDS = [1]
_TOL = 1e-6                  # relative tolerance for --verify value reconciliation


# --------------------------------------------------------------------------- #
# Problem rebuild (mirrors experiments.py:200-236; not exposed as a function
# there, so replicated here). Fetches prices live from yfinance, derives the
# moments, and returns the arrays needed to construct RingXYCardinalityQAOA.
# --------------------------------------------------------------------------- #
def prepare_problem(experiment_id, experiment):
    """Return (problem_dict, None) or (None, skip_reason)."""
    stocks = experiment["stocks"]
    start = experiment["start"]
    end = experiment["end"]
    budget = experiment["budget"]
    try:
        fetched = fetch_close_prices(
            stocks, start=start, end=end, auto_adjust=False, progress=True
        )
        close = fetched.close
        if close.shape[1] < 2:
            return None, "insufficient_valid_tickers"

        prices_now = close.iloc[-1]
        returns = close.pct_change(fill_method=None).dropna(how="any")
        stocks = returns.columns
        numpy_returns = returns.to_numpy()

        expected_returns = mean_historical_return(
            returns, returns_data=True, compounding=False
        ).to_numpy()
        covariance_matrix = sample_cov(returns, returns_data=True).to_numpy()
        coskewness_tensor = coskewness(numpy_returns)
        cokurtosis_tensor = cokurtosis(numpy_returns)
    except Exception as e:  # network / data hiccup -> skip this instance, never abort
        return None, f"fetch_or_moments_error: {e}"

    return {
        "experiment_id": experiment_id,
        "stocks": stocks,                       # pandas Index, like experiments.py
        "stock_names": [str(s) for s in stocks],
        "prices_now": prices_now,               # pd.Series indexed by ticker
        "expected_returns": expected_returns,
        "covariance_matrix": covariance_matrix,
        "coskewness_tensor": coskewness_tensor,
        "cokurtosis_tensor": cokurtosis_tensor,
        "budget": float(budget),
        "N": int(len(stocks)),
    }, None


def make_solver(problem, weighting, budget_override=None):
    """Construct a RingXYCardinalityQAOA from a rebuilt problem.

    ``budget_override`` is used only for the equal_split energy diagonal (a
    throwaway instance with budget/K). Allocation always uses the real budget.
    """
    return RingXYCardinalityQAOA(
        stocks=problem["stocks"],
        prices_now=problem["prices_now"],
        expected_returns=problem["expected_returns"],
        covariance_matrix=problem["covariance_matrix"],
        budget=(problem["budget"] if budget_override is None else budget_override),
        coskewness_tensor=problem["coskewness_tensor"],
        cokurtosis_tensor=problem["cokurtosis_tensor"],
        risk_aversion=RISK_AVERSION,
        selection_weighting=weighting,
    )


def decode_indices(z, N):
    """Subset indices for basis state z, identical to ring_xy.solve_selection_exactly.

    bitstring = format(z, f"0{N}b"); bit q == "1" means asset q selected. This is
    the exact convention the selection-energy diagonal d[z] is indexed by.
    """
    bitstring = format(int(z), f"0{N}b")
    return [q for q in range(N) if bitstring[q] == "1"]


# --------------------------------------------------------------------------- #
# Shared, weighting-independent allocator map: post_objective per subset.
# _allocate_on_subset never reads the selection Hamiltonian or the weighting, so
# this is computed ONCE per instance and reused across all three weightings.
# --------------------------------------------------------------------------- #
def build_allocator_map(solver, N, k_min):
    """z -> post_objective (float) for feasible subsets, None for infeasible.

    Covers every z with Hamming weight in [k_min, N].
    """
    post_map = {}
    for z in range(2 ** N):
        if bin(z).count("1") < k_min:
            continue
        sel_idx = decode_indices(z, N)
        alloc = solver._allocate_on_subset(sel_idx)
        post = alloc["post_objective"]
        post_map[z] = None if (alloc["infeasible"] or post is None) else float(post)
    return post_map


def energy_diagonal(solver, k_min):
    """Full 2**N diagonal of the (normalized) selection Hamiltonian for a solver."""
    solver.construct_selection_hubo_bin(k_min)
    return solver._selection_energy_diagonal()


def equal_split_diagonal(problem, K, k_min):
    """Equal-split (n_i = budget/(K*price_i)) energy diagonal at cardinality K.

    Reuses the exact budget-weighting code path on a throwaway instance with
    budget/K: its share counts become (budget/K)/price_i, so the diagonal entry
    for subset z is cost((n/K).z). The throwaway's own normalization factor differs
    from the real instance's, but a positive rescale is monotone, so WITHIN each
    fixed K the rank correlation is identical to the true equal-split ranking. The
    throwaway is used only to read this diagonal -- never to allocate.
    """
    throwaway = make_solver(problem, "budget", budget_override=problem["budget"] / K)
    return energy_diagonal(throwaway, k_min)


# --------------------------------------------------------------------------- #
# Correlation of (energy, post_objective) over feasible size-K subsets.
# --------------------------------------------------------------------------- #
def correlation_row(problem, K, weighting, diag, idxs, post_map, k_min):
    """Build one per-(experiment, N, K, weighting) record."""
    E, P = [], []
    for z in idxs:
        post = post_map[z]
        if post is None:                  # infeasible subset -> excluded
            continue
        E.append(float(diag[z]))
        P.append(float(post))

    n_subsets = len(idxs)                  # == math.comb(N, K)
    n_feasible = len(E)
    tau = rho = tau_p = rho_p = float("nan")
    if n_feasible >= 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # silence constant-input / tie warnings
            kt = kendalltau(E, P)             # tau-b: correct under frequent ties
            sr = spearmanr(E, P)             # midrank ties
        tau = float(kt.statistic)
        tau_p = float(kt.pvalue)
        rho = float(sr.statistic)
        rho_p = float(sr.pvalue)
    tau_oriented = (-tau) if not math.isnan(tau) else float("nan")

    return {
        "experiment_id": problem["experiment_id"],
        "N": problem["N"],
        "K": K,
        "weighting": weighting,
        "kendall_tau": tau,
        "tau_oriented": tau_oriented,
        "spearman_rho": rho,
        "kendall_p": tau_p,
        "spearman_p": rho_p,
        "n_subsets": n_subsets,
        "n_feasible": n_feasible,
        "budget": problem["budget"],
        "k_min": k_min,
        "stocks": ";".join(problem["stock_names"]),
    }


def audit_instance(problem, k_min, weightings, return_internals=False):
    """Compute per-(K, weighting) correlation rows for one problem instance."""
    N = problem["N"]
    k_min = max(1, min(int(k_min), N))

    # Shared allocator map + the budget solver (also reused for the budget diag).
    base = make_solver(problem, "budget")
    post_map = build_allocator_map(base, N, k_min)

    # K-independent diagonals computed once.
    diags = {}
    if "budget" in weightings:
        diags["budget"] = energy_diagonal(base, k_min)
    if "unit" in weightings:
        diags["unit"] = energy_diagonal(make_solver(problem, "unit"), k_min)

    rows = []
    for weighting in weightings:
        for K in range(k_min, N + 1):
            idxs = hamming_weight_indices(N, K)
            if weighting == "equal_split":
                diag = equal_split_diagonal(problem, K, k_min)   # per-K throwaway
            else:
                diag = diags[weighting]
            rows.append(
                correlation_row(problem, K, weighting, diag, idxs, post_map, k_min)
            )

    if return_internals:
        return rows, {"post_map": post_map, "base": base,
                      "budget_diag": diags.get("budget")}
    return rows


# --------------------------------------------------------------------------- #
# Aggregation into per-(N, K, weighting) cells.
# --------------------------------------------------------------------------- #
def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["N"], r["K"], r["weighting"])].append(r)

    out = []
    for (N, K, weighting), rs in sorted(groups.items()):
        taus = np.array([r["kendall_tau"] for r in rs], dtype=float)
        rhos = np.array([r["spearman_rho"] for r in rs], dtype=float)
        nfeas = np.array([r["n_feasible"] for r in rs], dtype=float)
        n_instances = len(rs)
        n_defined = int(np.sum(~np.isnan(taus)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # all-NaN slice -> NaN, no warning spam
            mean_tau = float(np.nanmean(taus)) if n_defined else float("nan")
            median_tau = float(np.nanmedian(taus)) if n_defined else float("nan")
            std_tau = float(np.nanstd(taus)) if n_defined else float("nan")
            mean_rho = float(np.nanmean(rhos)) if n_defined else float("nan")
            median_rho = float(np.nanmedian(rhos)) if n_defined else float("nan")
            std_rho = float(np.nanstd(rhos)) if n_defined else float("nan")
        out.append({
            "N": N,
            "K": K,
            "weighting": weighting,
            "mean_kendall_tau": mean_tau,
            "median_kendall_tau": median_tau,
            "std_kendall_tau": std_tau,
            "mean_spearman_rho": mean_rho,
            "median_spearman_rho": median_rho,
            "std_spearman_rho": std_rho,
            "n_instances": n_instances,
            "n_instances_defined": n_defined,
            "n_instances_nan": n_instances - n_defined,
            "mean_n_feasible": float(np.mean(nfeas)) if n_instances else float("nan"),
        })
    return out


# --------------------------------------------------------------------------- #
# Output writers (UTF-8 + newline="" to avoid Windows CRLF doubling).
# --------------------------------------------------------------------------- #
_PER_ROW_COLS = [
    "experiment_id", "N", "K", "weighting", "kendall_tau", "tau_oriented",
    "spearman_rho", "kendall_p", "spearman_p", "n_subsets", "n_feasible",
    "budget", "k_min", "stocks",
]
_AGG_COLS = [
    "N", "K", "weighting", "mean_kendall_tau", "median_kendall_tau",
    "std_kendall_tau", "mean_spearman_rho", "median_spearman_rho",
    "std_spearman_rho", "n_instances", "n_instances_defined", "n_instances_nan",
    "mean_n_feasible",
]


def write_csv(path, columns, records):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for rec in records:
            w.writerow({c: rec.get(c, "") for c in columns})
    print(f"  wrote {path} ({len(records)} rows)")


def make_figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Pool raw Kendall-tau by (weighting, K) across all instances / N.
    pooled = defaultdict(list)
    for r in rows:
        if not math.isnan(r["kendall_tau"]):
            pooled[(r["weighting"], r["K"])].append(r["kendall_tau"])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for weighting in WEIGHTINGS:
        ks = sorted({K for (w, K) in pooled if w == weighting})
        if not ks:
            continue
        ys = [float(np.mean(pooled[(weighting, K)])) for K in ks]
        ax.plot(ks, ys, marker="o", label=weighting)
    ax.axhline(0.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("cardinality K")
    ax.set_ylabel(r"mean Kendall $\tau$  (selection energy vs post_objective)")
    ax.set_title(
        "Selection-energy ranking quality vs cardinality\n"
        "energy lower=better, post_objective higher=better -> good ranking gives "
        r"$\tau<0$"
    )
    ax.legend(title="weighting")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# --------------------------------------------------------------------------- #
# --verify: correctness self-tests.
# --------------------------------------------------------------------------- #
def _find_row(rows, weighting, K):
    for r in rows:
        if r["weighting"] == weighting and r["K"] == K:
            return r
    return None


def _close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=_TOL, abs_tol=1e-9)


def run_verify(experiments, lo, hi, args):
    """Checks (a) sign at K=1 budget, (b) reconcile with solve_selection_exactly,
    (c) unit-weighting K=1 inversion surfaces."""
    a_pass = a_fail = a_skip = a_exact = 0
    b_instances = b_ok = 0
    b_cells = b_cells_ok = 0
    c_reports = []

    for eid in range(lo, hi + 1):
        problem, reason = prepare_problem(eid, experiments[eid])
        if problem is None:
            print(f"[skip] id {eid}: {reason}")
            continue
        if problem["N"] > args.enum_max_n:
            print(f"[skip] id {eid}: too_large (N={problem['N']})")
            continue
        N = problem["N"]
        k_min = max(1, min(int(args.k_min), N))
        print(f"[verify] id {eid}: N={N} budget={problem['budget']}")

        rows, internals = audit_instance(
            problem, k_min, ["budget", "unit"], return_internals=True
        )
        post_map = internals["post_map"]
        budget_diag = internals["budget_diag"]
        base = internals["base"]

        # (a) K=1 budget: a good ranking must give NEGATIVE Kendall-tau.
        ra = _find_row(rows, "budget", 1) if k_min <= 1 else None
        if ra is None or ra["n_feasible"] < 2 or math.isnan(ra["kendall_tau"]):
            a_skip += 1
        elif ra["kendall_tau"] < 0:
            a_pass += 1
            if math.isclose(abs(ra["kendall_tau"]), 1.0, abs_tol=1e-9):
                a_exact += 1
        else:
            a_fail += 1
            print(f"    check(a) FAIL id {eid}: budget K=1 tau={ra['kendall_tau']:.4f} (expected < 0)")

        # (b) Reconcile audit enumeration with the class's own exact reference,
        # computed LIVE on the same rebuilt problem (drift-immune; budget weighting).
        sse = base.solve_selection_exactly(
            k_min=k_min, enumerate_post_objective=True,
            enumeration_max_n=max(12, N),
        )
        e_perK = sse["energy_optimal"]["per_K"]
        p_perK = sse["post_objective_optimal"]["per_K"]
        inst_ok = True
        for K in range(k_min, N + 1):
            idxs = hamming_weight_indices(N, K)
            e_ref = e_perK[K - k_min]
            p_ref = p_perK[K - k_min]
            assert e_ref["K"] == K and p_ref["K"] == K, "per_K not ascending from k_min"

            # min selection energy over the size-K subspace (always defined).
            my_emin = float(min(budget_diag[z] for z in idxs))
            feas = [post_map[z] for z in idxs if post_map[z] is not None]
            my_pmax = (max(feas) if feas else None)
            my_nfeas = len(feas)

            cell_ok = (
                _close(my_emin, e_ref["selection_energy"])
                and _close(my_pmax, p_ref["post_objective"])
                and my_nfeas == int(p_ref["n_feasible_subsets"])
                and len(idxs) == int(p_ref["n_subsets"])
            )
            b_cells += 1
            if cell_ok:
                b_cells_ok += 1
            else:
                inst_ok = False
                print(f"    check(b) mismatch id {eid} K={K}: "
                      f"emin {my_emin:.6g}/{e_ref['selection_energy']:.6g} "
                      f"pmax {my_pmax}/{p_ref['post_objective']} "
                      f"nfeas {my_nfeas}/{p_ref['n_feasible_subsets']} "
                      f"nsub {len(idxs)}/{p_ref['n_subsets']}")
        b_instances += 1
        b_ok += int(inst_ok)

        # (c) unit-weighting K=1 inversion surfaces (soft, reported not asserted).
        if eid in INVERSION_IDS and k_min <= 1:
            rb = _find_row(rows, "budget", 1)
            ru = _find_row(rows, "unit", 1)
            if rb is not None and ru is not None:
                c_reports.append((eid, rb["kendall_tau"], ru["kendall_tau"]))

    print()
    print("==== verify summary ====")
    print(f"check (a) sign of budget K=1 tau (<0): "
          f"{a_pass} pass / {a_fail} fail / {a_skip} skip "
          f"({a_exact} of the passes are exactly |tau|=1)")
    print(f"check (b) reconcile vs solve_selection_exactly: "
          f"{b_ok}/{b_instances} instances fully match "
          f"({b_cells_ok}/{b_cells} per-K cells)")
    if c_reports:
        print("check (c) unit-weighting K=1 inversion (budget tau vs unit tau):")
        for eid, tb, tu in c_reports:
            note = "INVERTED (unit not <0)" if (math.isnan(tu) or tu >= 0) else "still negative"
            print(f"    id {eid}: budget={tb:.4f}  unit={tu:.4f}  -> {note}")
    else:
        print("check (c) unit-weighting K=1 inversion: no flagged ids in range")


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
def load_experiments(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(json.load(f)["data"])


def resolve_range(start, end, n):
    if start is None:
        return 0, n - 1
    end = end if end is not None else start
    if end < start:
        print(f"Error: invalid range start={start} end={end} (end must be >= start)")
        sys.exit(1)
    if not (0 <= start <= end <= n - 1):
        print(f"Error: range {start}..{end} out of bounds for {n} experiments (0..{n - 1})")
        sys.exit(1)
    return start, end


def main():
    parser = argparse.ArgumentParser(
        description="Rank-correlation audit: ring-XY selection energy vs "
        "post-allocation objective over all C(N,K) subsets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("start", type=int, nargs="?", default=None,
                        help="First experiment id (0-based, inclusive). Omit for all.")
    parser.add_argument("end", type=int, nargs="?", default=None,
                        help="Last experiment id (0-based, inclusive). Defaults to start.")
    parser.add_argument("--experiments-json", default="experiments_data.json")
    parser.add_argument("--weightings", nargs="+", default=list(WEIGHTINGS),
                        choices=WEIGHTINGS, help="Selection weightings to audit.")
    parser.add_argument("--k-min", type=int, default=1,
                        help="Smallest cardinality K (default 1; K=0 is excluded).")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--enum-max-n", type=int, default=ENUM_MAX_N,
                        help="Skip instances with N above this (2**N allocator calls).")
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--verify", action="store_true",
                        help="Run correctness self-tests instead of writing CSVs.")
    args = parser.parse_args()

    experiments = load_experiments(args.experiments_json)
    lo, hi = resolve_range(args.start, args.end, len(experiments))
    os.makedirs(args.out_dir, exist_ok=True)

    if args.verify:
        run_verify(experiments, lo, hi, args)
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rows = []
    skips = []
    for eid in range(lo, hi + 1):
        problem, reason = prepare_problem(eid, experiments[eid])
        if problem is None:
            print(f"[skip] id {eid}: {reason}")
            skips.append({"experiment_id": eid, "reason": reason})
            continue
        if problem["N"] > args.enum_max_n:
            print(f"[skip] id {eid}: too_large (N={problem['N']} > {args.enum_max_n})")
            skips.append({"experiment_id": eid, "reason": f"too_large_N={problem['N']}"})
            continue
        print(f"[audit] id {eid}: N={problem['N']} budget={problem['budget']}")
        all_rows.extend(audit_instance(problem, args.k_min, args.weightings))

    print("\n==== writing artifacts ====")
    per_row_path = os.path.join(
        args.out_dir, f"rank_correlation_audit_per_row_{timestamp}.csv")
    agg_path = os.path.join(
        args.out_dir, f"rank_correlation_audit_aggregated_{timestamp}.csv")
    write_csv(per_row_path, _PER_ROW_COLS, all_rows)
    write_csv(agg_path, _AGG_COLS, aggregate(all_rows))
    if skips:
        skip_path = os.path.join(
            args.out_dir, f"rank_correlation_audit_skips_{timestamp}.csv")
        write_csv(skip_path, ["experiment_id", "reason"], skips)
    if not args.no_figure and all_rows:
        make_figure(all_rows, os.path.join(
            args.out_dir, f"rank_correlation_audit_{timestamp}.png"))

    n_inst = len({(r["experiment_id"]) for r in all_rows})
    print(f"\nDone: {len(all_rows)} rows over {n_inst} instances, {len(skips)} skipped.")
    print("Reminder: energy lower=better, post_objective higher=better -> a good "
          "selection ranking gives NEGATIVE kendall_tau (tau_oriented = -kendall_tau).")


if __name__ == "__main__":
    main()
