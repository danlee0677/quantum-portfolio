"""Profile / benchmarking tool for the HUBO and cardinality QAOA pipelines.

Subcommands:
  specs        — static circuit metrics (qubits, depth, gate counts) on all 100
                 problems; cheap.
  convergence  — CMA-ES dynamics (evaluations, wall-clock, approx ratio) on a
                 small subset; expensive.

Method selection:
  --method {hopo,ring_xy,both}  — emit integer-HUBO rows, cardinality rows, or
  both (default). Integer-HUBO rows come from HigherOrderPortfolioQAOA
  (portfolio_hubo_qaoa_light.py); cardinality rows come from
  RingXYCardinalityQAOA (ring_xy.py).

Experiment range:
  Each subcommand takes optional positional `start end` (0-based, inclusive),
  e.g. `profile.py specs 1 4`. Omit them to profile all problems.

Best-K resolution (cardinality only):
  default → read cardinality_qaoa_solution.best_K, pulling each requested
  experiment from the newest results/ring_xy_exp_*.json file that contains it
  (per-experiment merge, newest-first). An explicit --results-json uses that one
  file. --rederive-k or no results JSON found → run solve_with_qaoa_cardinality
  on every K to find the best.

Cardinality row source (convergence only):
  --card-source {live,json} — 'live' (default) re-runs the fixed-K cardinality
  QAOA; 'json' loads the best_K per_K entry from the same merged ring_xy results
  stream used for best_K resolution, falling back to a live run per experiment
  when the entry is missing or unusable (mirrors --hopo-source).
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import pennylane as qml
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov

from coskweness_cokurtosis import cokurtosis, coskewness
from data_provider import PriceCache, fetch_close_prices
from portfolio_hubo_qaoa_light import HigherOrderPortfolioQAOA
from ring_xy import RingXYCardinalityQAOA
from utils import decompose_ry, replace_h_rz_h_with_rx

HYPERPARAMS = {
    "risk_aversion": 0.1,
    "max_qubits": 15,
    "log_encoding": True,
    "strict_budget_constraint": False,
    "lambda_budget": 0.001,
}

ALLOWED_GATES = ["CNOT", "RZ", "RX", "Hadamard"]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def load_problems(path):
    with open(path, "r") as f:
        return list(json.load(f)["data"])


RESULTS_DIR = "results"
_TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")


def _results_glob(method):
    # best_K is read only from the cardinality (ring_xy) result stream.
    if method in ("ring_xy", "both"):
        return os.path.join(RESULTS_DIR, "ring_xy_exp_*.json")
    return os.path.join(RESULTS_DIR, "portfolio_optimization_exp_*.json")


def _file_sort_key(path):
    """Newest-first ordering key: parse the YYYYMMDD_HHMMSS stamp from the
    filename, falling back to file mtime when it is absent."""
    m = _TIMESTAMP_RE.search(os.path.basename(path))
    if m:
        return (1, m.group(1))
    try:
        return (0, str(os.path.getmtime(path)))
    except OSError:
        return (0, "")


def _merge_files_by_id(files, id_range):
    """Pull each requested experiment from the newest file that contains it.

    files must already be sorted newest-first. id_range is an inclusive
    (start, end) tuple, or None to take everything. Returns (merged, provenance,
    files_used) where provenance maps the string experiment id to the file it was
    pulled from and files_used is the newest-first list of files actually opened.
    """
    needed = set(range(id_range[0], id_range[1] + 1)) if id_range is not None else None

    merged = {}
    provenance = {}
    files_used = []
    for path in files:
        if needed is not None and not needed - set(int(k) for k in merged):
            break  # every requested id already covered by a newer file
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  skipping unreadable results file {path}: {e}")
            continue
        opened = False
        for key, entry in data.items():
            if key in merged:
                continue
            try:
                idx = int(key)
            except ValueError:
                continue
            if needed is not None and idx not in needed:
                continue
            merged[key] = entry
            provenance[key] = path
            opened = True
        if opened:
            files_used.append(path)
    return merged, provenance, files_used


def build_merged_results(method, id_range):
    """Assemble the ring_xy results dict (best_K stream) by pulling each
    experiment from the newest file that contains it."""
    files = sorted(glob.glob(_results_glob(method)), key=_file_sort_key, reverse=True)
    return _merge_files_by_id(files, id_range)


def build_hopo_results(id_range, explicit_path=None):
    """Assemble the HOPO-with-QAOA results dict for --hopo-source json.

    Independent of the ring_xy best_K stream so --method both can resolve each
    from its own files. An explicit_path uses that single current-format file
    as-is; otherwise pull each requested experiment from the newest
    results/portfolio_optimization_exp_*.json that contains it.
    """
    if explicit_path is not None:
        data = load_results(explicit_path)
        if data is None:
            return {}, {}, []
        merged, _, _ = _merge_files_by_id([explicit_path], id_range)
        provenance = {k: explicit_path for k in merged}
        return merged, provenance, ([explicit_path] if merged else [])
    pattern = os.path.join(RESULTS_DIR, "portfolio_optimization_exp_*.json")
    files = sorted(glob.glob(pattern), key=_file_sort_key, reverse=True)
    return _merge_files_by_id(files, id_range)


def load_results(path):
    if path is None or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def prepare_problem_data(experiment, cache=None):
    """Download + derive the moments shared by both QAOA pipelines.

    When ``cache`` (a ``PriceCache``) is supplied, per-ticker price series are
    fetched at most once across the run; with ``cache=None`` the behavior is the
    original per-problem download.
    """
    stocks = experiment["stocks"]
    start = experiment["start"]
    end = experiment["end"]
    budget = experiment["budget"]
    # Retry-aware fetch: transient Yahoo throttling is retried, only genuinely
    # dead tickers are dropped (matches experiments.py; auto_adjust=True here).
    if cache is not None:
        close = cache.get_close(stocks, start=start, end=end, auto_adjust=True)
    else:
        close = fetch_close_prices(
            stocks, start=start, end=end, auto_adjust=True, progress=False
        ).close
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
    return {
        "stocks": stocks,
        "prices_now": prices_now,
        "expected_returns": expected_returns,
        "covariance_matrix": covariance_matrix,
        "coskewness_tensor": coskewness_tensor,
        "cokurtosis_tensor": cokurtosis_tensor,
        "budget": budget,
    }


def build_portfolio_hubo(experiment, data=None):
    d = data if data is not None else prepare_problem_data(experiment)
    return HigherOrderPortfolioQAOA(
        stocks=d["stocks"],
        prices_now=d["prices_now"],
        expected_returns=d["expected_returns"],
        covariance_matrix=d["covariance_matrix"],
        budget=d["budget"],
        coskewness_tensor=d["coskewness_tensor"],
        cokurtosis_tensor=d["cokurtosis_tensor"],
        **HYPERPARAMS,
    )


def build_ring_xy(experiment, data=None, selection_weighting="budget"):
    d = data if data is not None else prepare_problem_data(experiment)
    return RingXYCardinalityQAOA(
        stocks=d["stocks"],
        prices_now=d["prices_now"],
        expected_returns=d["expected_returns"],
        covariance_matrix=d["covariance_matrix"],
        budget=d["budget"],
        coskewness_tensor=d["coskewness_tensor"],
        cokurtosis_tensor=d["cokurtosis_tensor"],
        risk_aversion=HYPERPARAMS["risk_aversion"],
        selection_weighting=selection_weighting,
    )


def _rederive_best_K(ring_xy_solver, k_min=1):
    N = ring_xy_solver.num_assets
    best_obj = None
    best_K = None
    for K in range(k_min, N + 1):
        try:
            res = ring_xy_solver.solve_with_qaoa_cardinality(K)
        except Exception as e:
            print(f"  re-derive K={K} failed: {e}")
            continue
        if res["infeasible"] or res["post_objective"] is None:
            continue
        if best_obj is None or res["post_objective"] > best_obj:
            best_obj = res["post_objective"]
            best_K = K
    return best_K


def _recorded_no_feasible_K(entry):
    """True when the entry's cardinality_qaoa_solution explicitly records
    best_K=null (the experiments run found no feasible K), as opposed to
    lacking the key (old/foreign file, where re-deriving is appropriate)."""
    card = (entry or {}).get("cardinality_qaoa_solution")
    return isinstance(card, dict) and "best_K" in card and card["best_K"] is None


def resolve_best_K(ring_xy_solver, problem_id, results, rederive, k_min=1):
    if rederive or results is None:
        return _rederive_best_K(ring_xy_solver, k_min=k_min)
    entry = results.get(str(problem_id))
    if entry is None:
        print(
            f"  results JSON has no entry for problem {problem_id}; re-deriving best_K"
        )
        return _rederive_best_K(ring_xy_solver, k_min=k_min)
    if _recorded_no_feasible_K(entry):
        print(
            f"  problem {problem_id}: results JSON records best_K=null "
            f"(no feasible K); skipping (use --rederive-k to force a sweep)"
        )
        return None
    card = entry.get("cardinality_qaoa_solution")
    best_K = card.get("best_K") if isinstance(card, dict) else None
    if best_K is None:
        print(f"  results JSON missing best_K for problem {problem_id}; re-deriving")
        return _rederive_best_K(ring_xy_solver, k_min=k_min)
    return int(best_K)


# --------------------------------------------------------------------------- #
# Fig. 6 comparison: budget utilization + continuous-baseline / exact-reference
# enrichment. The portfolio objective f(z) (get_objective_value) carries no
# budget term, so it scales with deployed capital and is a valid common currency
# ONLY among budget-feasible allocations: continuous & ring-XY are LP-capped
# (budget_util <= 1), but integer-HUBO uses a soft penalty and can overspend
# (budget_util > 1). Every series therefore carries budget_util so the consumer
# can gate feasibility; over_budget flags the violators explicitly.
# --------------------------------------------------------------------------- #
BUDGET_TOL = 1e-6  # budget_util <= 1 + BUDGET_TOL counts as feasible (absorbs rounding)


def _budget_util(realized, total_budget):
    """(budget_util, over_budget) = (realized/total, util > 1+tol), or (None, None)
    when unrecoverable. Used for the LP-capped series (continuous, ring-XY, ref)."""
    try:
        if realized is None or not total_budget:
            return None, None
        util = float(realized) / float(total_budget)
    except (TypeError, ValueError, ZeroDivisionError):
        return None, None
    return util, util > 1.0 + BUDGET_TOL


def _hopo_budget_util(obj_vals, result_with_budget, total_budget):
    """Realized budget util for the integer-HUBO solution matching max(obj_vals).

    The two-most-probable-states ordering is shared across objective_values,
    result_with_budget, and optimized_portfolios (built by solve_with_qaoa_cma_es
    in portfolio_hubo_qaoa_light.py, from two_most_probable = argsort(probs)[-2:]),
    so the post_objective (= max obj_val) and its realized budget come from the
    same argmax index. result_with_budget[i]["budget"] is the amount *spent*,
    which can exceed total_budget (soft penalty only) -> over_budget True.
    Returns (None, None) when unrecoverable."""
    try:
        if not obj_vals or not result_with_budget or not total_budget:
            return None, None
        idx = max(range(len(obj_vals)), key=lambda i: obj_vals[i])
        if idx >= len(result_with_budget):
            return None, None
        spent = result_with_budget[idx].get("budget")
    except (TypeError, ValueError, AttributeError, KeyError):
        return None, None
    return _budget_util(spent, total_budget)


# --------------------------------------------------------------------------- #
# Price-weighted expected-return ("profit") helpers for the benefit-vs-N view.
# The HUBO objective's return term is price-blind (Σ shares·rate, not money), so a
# real profit proxy must re-weight by current price: Σ shares·price·rate has units
# dollars/yr. μ (expected_returns, an annualized rate) and prices_now are NOT saved
# in the results JSON, so they are recomputed cheaply (no QAOA) from the saved
# hyperparams.{stocks,start,end} — and in a live run the per-run PriceCache returns
# the exact same series the live solve used (identical dates -> identical μ).
# --------------------------------------------------------------------------- #
def _ticker_return_maps(stocks, expected_returns, prices_now):
    """(mu_by_ticker, price_by_ticker) aligned by TICKER NAME, never by position.

    ``expected_returns`` is positionally aligned to ``stocks``; ``prices_now`` is a
    ticker-indexed pandas Series. Returns (None, None) on any failure."""
    try:
        mu = {str(s): float(m) for s, m in zip(list(stocks), list(expected_returns))}
        price = {str(t): float(prices_now[t]) for t in prices_now.index}
    except (TypeError, ValueError, KeyError, IndexError):
        return None, None
    return mu, price


def _recompute_return_maps(hp, cache):
    """Cheap μ + price recompute (no solver build) from saved hyperparams, for the
    benefit-vs-N profit term. Refetches closes for ``hp.{stocks,start,end}`` through
    the shared ``cache`` (a live run already cached these, so it is a cache hit and
    the μ matches the live solve exactly). Returns (None, None) on any failure."""
    stocks = hp.get("stocks")
    start = hp.get("start")
    end = hp.get("end")
    if not stocks or not start or not end or cache is None:
        return None, None
    try:
        close = cache.get_close(stocks, start=start, end=end, auto_adjust=True)
        if close is None or close.empty:
            return None, None
        prices_now = close.iloc[-1]
        returns = close.pct_change(fill_method=None).dropna(how="any")
        cols = list(returns.columns)
        expected_returns = mean_historical_return(
            returns, returns_data=True, compounding=False
        ).to_numpy()
        return _ticker_return_maps(cols, expected_returns, prices_now)
    except Exception as e:
        print(f"  μ/price recompute failed: {e}")
        return None, None


def _expected_return_dollars(allocation, mu_by_ticker, price_by_ticker):
    """Price-weighted expected annual return (dollars/yr) of a share allocation:
    ``Σ_t shares_t · price_t · μ_t`` over tickers held in ``allocation``. Returns
    None if the maps are missing. Tickers held but absent from either map are
    skipped and logged (never index-zipped against the wrong asset)."""
    if not isinstance(allocation, dict) or not mu_by_ticker or not price_by_ticker:
        return None
    total = 0.0
    missing = []
    for tkr, shares in allocation.items():
        key = str(tkr)
        if key in mu_by_ticker and key in price_by_ticker:
            try:
                total += float(shares) * float(price_by_ticker[key]) * float(
                    mu_by_ticker[key]
                )
            except (TypeError, ValueError):
                if shares:
                    missing.append(key)
        elif shares:  # only flag tickers actually held
            missing.append(key)
    if missing:
        print(
            f"  return calc: skipped held tickers absent from μ/price maps: "
            f"{sorted(set(missing))}"
        )
    return total


def compare_enrichment(entry, variant="constrained", mu_by_ticker=None,
                       price_by_ticker=None):
    """Per-problem continuous-baseline + exact-reference fields for the Fig. 6
    comparison. All values None-safe; a missing/aberrant entry yields all-None.

    ``entry`` is a results-JSON experiment dict — prefer the ring_xy stream,
    which also carries ``selection_exact_solution`` (the HOPO stream lacks it, so
    ``reference_*`` stays None there). ``variant`` selects the continuous series:
    'constrained' (always stored) is the primary baseline; 'unconstrained' uses
    the penalty variant (Eq. 19, None without higher moments) as the primary;
    'both' keeps constrained primary and emits the unconstrained series too.

    When ``mu_by_ticker``/``price_by_ticker`` are supplied, also emits the
    price-weighted expected-return ("profit") fields for the benefit-vs-N view:
    ``continuous_return`` (always plotted) and ``reference_return`` (reserved,
    not plotted in v1). Both are dollars/yr; left None when the maps or the
    corresponding allocation are unavailable.
    """
    out = {
        "total_budget": None,
        "continuous_objective": None,
        "continuous_budget_util": None,
        "continuous_unconstrained_objective": None,
        "continuous_unconstrained_budget_util": None,
        "reference_objective": None,
        "reference_budget_util": None,
        "continuous_return": None,
        "reference_return": None,
    }
    if not isinstance(entry, dict) or "error" in entry:
        return out

    hp = entry.get("hyperparams") or {}
    try:
        out["total_budget"] = (
            float(hp["budget"]) if hp.get("budget") is not None else None
        )
    except (TypeError, ValueError):
        out["total_budget"] = None
    total = out["total_budget"]

    def _cont(key):
        """(objective, budget_util) for a stored continuous solution, or (None, None).
        realized = total_budget - left_overs (left_overs is leftover *cash*)."""
        sol = entry.get(key)
        if not isinstance(sol, dict):
            return None, None
        try:
            obj = float(sol["value"]) if sol.get("value") is not None else None
        except (TypeError, ValueError):
            obj = None
        realized = None
        if total is not None and sol.get("left_overs") is not None:
            try:
                realized = total - float(sol["left_overs"])
            except (TypeError, ValueError):
                realized = None
        util, _ = _budget_util(realized, total)
        return obj, util

    primary_key = (
        "continuous_variables_solution_unconstrained"
        if variant == "unconstrained"
        else "continuous_variables_solution"
    )
    out["continuous_objective"], out["continuous_budget_util"] = _cont(primary_key)
    if variant == "both":
        (
            out["continuous_unconstrained_objective"],
            out["continuous_unconstrained_budget_util"],
        ) = _cont("continuous_variables_solution_unconstrained")

    # Price-weighted expected-return ("profit") of the continuous allocation, in
    # dollars/yr, when the μ/price maps are supplied. The continuous allocation is
    # the LP-deterministic baseline portfolio (same source as continuous_objective).
    if mu_by_ticker and price_by_ticker:
        cont_sol = entry.get(primary_key)
        if isinstance(cont_sol, dict):
            out["continuous_return"] = _expected_return_dollars(
                cont_sol.get("allocation"), mu_by_ticker, price_by_ticker
            )

    # Exact reference (ring_xy only). Guard the full chain: the error path stores
    # {"error": ...} with no post_objective_optimal, and the non-enumerated path
    # leaves post_objective_optimal = None.
    selx = entry.get("selection_exact_solution")
    if isinstance(selx, dict) and "error" not in selx:
        poo = selx.get("post_objective_optimal")
        if isinstance(poo, dict):
            ref = poo.get("reference")
            if isinstance(ref, dict):
                try:
                    out["reference_objective"] = (
                        float(ref["post_objective"])
                        if ref.get("post_objective") is not None
                        else None
                    )
                except (TypeError, ValueError):
                    out["reference_objective"] = None
                out["reference_budget_util"], _ = _budget_util(
                    ref.get("realized_budget"), total
                )
                # Reserved (not plotted in v1): price-weighted return of the
                # allocator-optimal reference portfolio.
                if mu_by_ticker and price_by_ticker:
                    out["reference_return"] = _expected_return_dollars(
                        ref.get("allocation"), mu_by_ticker, price_by_ticker
                    )
    return out


def hopo_row_from_json(pid, entry):
    """Build an integer-HUBO convergence row from a saved HOPO entry instead of
    recomputing it live. Returns None when the entry lacks a usable
    qaoa_solution so the caller can fall back to a live solve.

    Mirrors the live row built in run_convergence (the want_hopo branch): same
    keys, with E_min/E_max derived from the saved exact spectrum, an added
    "source": "json" marker, and wall_clock_seconds=None (not recoverable).
    """
    if not isinstance(entry, dict):
        return None
    qaoa = entry.get("qaoa_solution")
    if (
        not isinstance(qaoa, dict)
        or "error" in entry
        or "final_expectation_value" not in qaoa
    ):
        return None

    fin_exp = float(qaoa["final_expectation_value"])
    obj_vals = qaoa.get("objective_values") or []
    post_objective = float(max(obj_vals)) if obj_vals else None
    hist = qaoa.get("training_history") or {}

    hp = entry.get("hyperparams") or {}
    n_qubits = hp.get("n_qubits")
    # Problem size is the asset count, not the log-encoding qubit count
    # (n_qubits > n_assets for integer-HUBO); no qubit fallback here.
    n_assets = len(hp.get("stocks") or []) or None

    # E_min/E_max from the saved exact spectrum. The lobpcg fallback at >=14
    # qubits stores only its few smallest eigenvalues under "spectrum", so
    # max(spectrum) is not the true spectral maximum and the min-max ratio
    # would explode; treat the spectrum as full only when its length matches
    # 2^n_qubits (newer files also carry an explicit spectrum_is_partial flag).
    exact = entry.get("exact_solution") or {}
    spectrum = exact.get("spectrum")
    E_min, E_max = None, None
    spectrum_partial = False
    if spectrum:
        flagged_partial = exact.get("spectrum_is_partial")
        spectrum_full = (flagged_partial is False) or (
            flagged_partial is None
            and n_qubits is not None
            and len(spectrum) == 2 ** int(n_qubits)
        )
        E_min = float(min(spectrum))
        if spectrum_full:
            E_max = float(max(spectrum))
        else:
            spectrum_partial = True
            print(
                f"  problem {pid}: partial exact spectrum ({len(spectrum)} of "
                f"2^{n_qubits} states); approximation ratio unavailable"
            )
    else:
        eigs = exact.get("smallest_eigenvalues")
        if eigs:
            E_min = float(eigs[0])

    ar = None
    if E_min is not None and E_max is not None and E_max > E_min:
        # min-max ratio, 0 = optimal; profile_visualize plots 1 - this
        ar = float((fin_exp - E_min) / (E_max - E_min))
    if ar is not None and not (-0.01 <= ar <= 1.01):
        print(
            f"  problem {pid}: approximation ratio {ar:.4g} outside [0, 1]; "
            f"dropping as corrupt"
        )
        ar = None

    budget_util, over_budget = _hopo_budget_util(
        obj_vals, qaoa.get("result_with_budget"), hp.get("budget")
    )

    return {
        "problem_id": pid,
        "method": "integer_hubo",
        "K": None,
        "n_qubits": int(n_qubits) if n_qubits is not None else None,
        "n_assets": n_assets,
        "source": "json",
        "wall_clock_seconds": None,
        "evaluations": int(hist.get("evaluations", 0)),
        "iterations": int(hist.get("iterations", 0)),
        "final_expectation_value": fin_exp,
        "post_objective": post_objective,
        "approximation_ratio": ar,
        "E_min": E_min,
        "E_max": E_max,
        "spectrum_partial": spectrum_partial,
        "budget_util": budget_util,
        "over_budget": over_budget,
        "training_history": hist,
    }


def _card_global_ar(g_min, g_max, final_expectation_value, pid=None):
    """Reference-derived global min-max approximation ratio (0 = optimal), or None.

    Mirrors the integer-HUBO (fin - E_min)/(E_max - E_min) block in
    hopo_row_from_json: same formula, same -0.01..1.01 corrupt guard. Returns
    None when the bounds are unusable (missing / non-finite / degenerate) or the
    ratio falls outside the guard band.
    """
    if g_min is None or g_max is None:
        return None
    g_min = float(g_min)
    g_max = float(g_max)
    if not (np.isfinite(g_min) and np.isfinite(g_max)) or g_max <= g_min:
        return None
    ar = float((float(final_expectation_value) - g_min) / (g_max - g_min))
    if not (-0.01 <= ar <= 1.01):
        print(
            f"  problem {pid}: approximation ratio {ar:.4g} outside [0, 1]; "
            f"dropping as corrupt"
        )
        return None
    return ar


def card_ar_from_reference(selection_exact, final_expectation_value, K, pid=None):
    """Compute the ring-XY approximation ratio from the stored exact selection
    reference (top-level ``selection_exact_solution``), symmetric to how
    hopo_row_from_json derives the integer-HUBO ratio from ``exact_solution``.

    Returns a dict ``{"usable": bool, ...}``. ``usable`` is False when the
    reference is absent / not a dict / carries an ``"error"`` key /
    ``spectrum_is_partial`` is True / the global min-max bounds are missing,
    non-finite, or degenerate — in which case the caller falls back to the
    QAOA-embedded ratio. When usable, the global ratio uses the K-independent
    top-level bounds; the diagnostic subspace fields use the per-K entry in
    ``energy_optimal.per_K`` with ``K == K`` (absent → left None, the global
    ratio is still returned).
    """
    if not isinstance(selection_exact, dict) or "error" in selection_exact:
        return {"usable": False}
    if selection_exact.get("spectrum_is_partial"):
        return {"usable": False}
    g_min = selection_exact.get("selection_energy_global_min")
    g_max = selection_exact.get("selection_energy_global_max")
    if g_min is None or g_max is None:
        return {"usable": False}
    g_min = float(g_min)
    g_max = float(g_max)
    if not (np.isfinite(g_min) and np.isfinite(g_max)) or g_max <= g_min:
        return {"usable": False}

    final = float(final_expectation_value)
    ar_global = _card_global_ar(g_min, g_max, final, pid)

    # Diagnostic subspace fields from the per-K energy-optimal reference (K-keyed).
    e_sub_min = e_sub_max = e_dicke_mean = ar_subspace = None
    per_K = ((selection_exact.get("energy_optimal") or {}).get("per_K")) or []
    pk = next((p for p in per_K if isinstance(p, dict) and p.get("K") == K), None)
    if pk is not None:
        e_sub_min = pk.get("selection_energy_min_subspace")
        e_sub_max = pk.get("selection_energy_max_subspace")
        e_dicke_mean = pk.get("selection_energy_dicke_mean")
        if e_sub_min is not None and e_dicke_mean is not None:
            denom = float(e_sub_min) - float(e_dicke_mean)
            if abs(denom) > 1e-12:
                ar_subspace = float((final - float(e_dicke_mean)) / denom)

    return {
        "usable": True,
        "approximation_ratio_global": ar_global,
        "E_global_min": g_min,
        "E_global_max": g_max,
        "approximation_ratio_subspace": ar_subspace,
        "E_subspace_min": e_sub_min,
        "E_subspace_max": e_sub_max,
        "E_dicke_mean": e_dicke_mean,
    }


def card_row_from_json(pid, entry, n_qubits):
    """Build a cardinality convergence row from a saved ring_xy entry instead of
    re-running the fixed-K QAOA. Returns None when the entry lacks a usable
    best_K per_K entry so the caller can fall back to a live solve. (An entry
    that explicitly records best_K=null never reaches this function — the
    caller detects it via _recorded_no_feasible_K and skips the problem
    outright instead of falling back to live.)

    Mirrors the live row built in run_convergence (the want_ring_xy branch):
    same keys, plus a "source": "json" marker and wall_clock_seconds=None (the
    current ring_xy JSON carries no per-K timing).

    The cardinality circuit uses one qubit per asset, so the row's n_qubits is
    num_assets (derived here from the entry's hyperparams.stocks), NOT the
    passed-in `n_qubits` — in --method both the latter is the integer-HUBO
    log-encoding count and can exceed num_assets. The passed-in value is kept
    only as a fallback when the entry lacks a stocks list.
    """
    if not isinstance(entry, dict) or "error" in entry:
        return None
    card = entry.get("cardinality_qaoa_solution")
    if not isinstance(card, dict):
        return None
    best_K = card.get("best_K")
    if best_K is None:
        return None
    e = next(
        (
            p
            for p in (card.get("per_K") or [])
            if isinstance(p, dict)
            and p.get("K") == best_K
            and "error" not in p
            and "final_expectation_value" in p
        ),
        None,
    )
    if e is None:
        return None

    num_assets = len((entry.get("hyperparams") or {}).get("stocks") or []) or n_qubits

    final_exp = float(e["final_expectation_value"])
    # Derive the global min-max ratio from the first-class exact reference
    # (selection_exact_solution), symmetric to how the integer-HUBO row derives
    # its ratio from exact_solution.spectrum. Fall back to the QAOA-embedded
    # value for older files that predate the reference (see card_ar_from_reference).
    ref = card_ar_from_reference(
        entry.get("selection_exact_solution"), final_exp, best_K, pid
    )
    if ref.get("usable"):
        ar_global = ref["approximation_ratio_global"]
        e_global_min, e_global_max = ref["E_global_min"], ref["E_global_max"]
        # Subspace diagnostics: prefer the reference per-K entry, else keep the
        # embedded values (best_K may be absent from the reference's per-K).
        ar_subspace = (
            ref["approximation_ratio_subspace"]
            if ref.get("approximation_ratio_subspace") is not None
            else e.get("approximation_ratio_subspace")
        )
        e_sub_min = (
            ref["E_subspace_min"]
            if ref.get("E_subspace_min") is not None
            else e.get("selection_energy_min_subspace")
        )
        e_sub_max = (
            ref["E_subspace_max"]
            if ref.get("E_subspace_max") is not None
            else e.get("selection_energy_max_subspace")
        )
        e_dicke_mean = (
            ref["E_dicke_mean"]
            if ref.get("E_dicke_mean") is not None
            else e.get("selection_energy_dicke_mean")
        )
        ar_source = "reference"
    else:
        print(
            f"  problem {pid}: no usable selection_exact_solution; using "
            f"QAOA-embedded approximation ratio"
        )
        ar_global = e.get("approximation_ratio_global")
        ar_subspace = e.get("approximation_ratio_subspace")
        e_sub_min = e.get("selection_energy_min_subspace")
        e_sub_max = e.get("selection_energy_max_subspace")
        e_dicke_mean = e.get("selection_energy_dicke_mean")
        e_global_min = e.get("selection_energy_global_min")
        e_global_max = e.get("selection_energy_global_max")
        ar_source = "embedded"

    budget_util, over_budget = _budget_util(
        None if e.get("infeasible") else e.get("realized_budget"),
        (entry.get("hyperparams") or {}).get("budget"),
    )

    return {
        "problem_id": pid,
        "method": "cardinality",
        "K": int(best_K),
        "n_qubits": num_assets,
        "n_assets": num_assets,
        "source": "json",
        "wall_clock_seconds": None,
        "evaluations": int((e.get("training_history") or {}).get("evaluations", 0)),
        "iterations": int(e.get("iterations", 0)),
        "final_expectation_value": final_exp,
        "approximation_ratio_subspace": ar_subspace,
        "approximation_ratio_global": ar_global,
        "approximation_ratio_global_source": ar_source,
        "E_subspace_min": e_sub_min,
        "E_subspace_max": e_sub_max,
        "E_dicke_mean": e_dicke_mean,
        "E_global_min": e_global_min,
        "E_global_max": e_global_max,
        "post_objective": e.get("post_objective"),
        "infeasible": e.get("infeasible"),
        "budget_util": budget_util,
        "over_budget": over_budget,
        # Carry the saved best_K allocation so the profit term (benefit-vs-N) is
        # computed from this row's OWN portfolio (Gap-1 coherence with post_objective).
        "allocation": e.get("allocation"),
        "training_history": e.get("training_history") or {},
    }


def compile_for_specs(circuit):
    # RY (from the Dicke StatePrep's Mottonen decomposition) has no legacy
    # decomposition, so the basis_set pass keeps it; rewrite it into the
    # allowed basis afterwards so both methods report in CNOT/RZ/RX/Hadamard.
    dispatched = qml.transform(replace_h_rz_h_with_rx)
    ry_to_zxz = qml.transform(decompose_ry)
    compiled = qml.compile(circuit, basis_set=ALLOWED_GATES)
    compiled = qml.compile(compiled, pipeline=[dispatched, ry_to_zxz])
    return compiled


def collect_specs(circuit, dummy_params):
    compiled = compile_for_specs(circuit)
    spec = qml.specs(compiled)(dummy_params)
    resources = spec.get("resources", None)
    if resources is not None:
        gate_types = dict(getattr(resources, "gate_types", {}))
        gate_sizes = dict(getattr(resources, "gate_sizes", {}))
        depth = int(getattr(resources, "depth", 0))
        total_gates = int(getattr(resources, "num_gates", 0))
    else:
        gate_types = dict(spec.get("gate_types", {}))
        gate_sizes = dict(spec.get("gate_sizes", {}))
        depth = int(spec.get("depth", 0))
        total_gates = int(spec.get("num_operations", spec.get("num_gates", 0)))
    two_qubit = int(gate_sizes.get(2, gate_sizes.get("2", 0)))
    unexpected = sorted(
        set(gate_types) - set(ALLOWED_GATES) - {"Identity", "GlobalPhase"}
    )
    return {
        "gate_counts": {g: int(c) for g, c in gate_types.items()},
        "gate_sizes": {str(k): int(v) for k, v in gate_sizes.items()},
        "two_qubit_gate_count": two_qubit,
        "depth": depth,
        "total_gates": total_gates,
        "unexpected_gates": unexpected,
    }


def run_specs(args, problems, results):
    rows = []
    cache = PriceCache()  # per-run price cache; each ticker downloaded at most once
    for pid, experiment in enumerate(problems):
        if args.id_range is not None and not (
            args.id_range[0] <= pid <= args.id_range[1]
        ):
            continue
        print(f"[specs] problem {pid} stocks={len(experiment['stocks'])}")
        try:
            data = prepare_problem_data(experiment, cache)
        except Exception as e:
            print(f"  build failed: {e}")
            rows.append({"problem_id": pid, "error": f"build: {e}"})
            continue

        # Integer-HUBO (original paper method)
        if args.method in ("hopo", "both"):
            portfolio_hubo = build_portfolio_hubo(experiment, data)
            N = portfolio_hubo.num_assets
            qcirc, _, layers_int, n_q_int = portfolio_hubo.build_integer_hubo_circuit(
                device_name="default.qubit"
            )
            dummy_int = np.zeros(2 * layers_int)
            try:
                int_spec = collect_specs(qcirc, dummy_int)
            except Exception as e:
                print(f"  integer-HUBO specs failed: {e}")
                int_spec = {"error": str(e)}
            rows.append(
                {
                    "problem_id": pid,
                    "n_assets": N,
                    "method": "integer_hubo",
                    "K": None,
                    "n_qubits": n_q_int,
                    "layers": layers_int,
                    "num_params": 2 * layers_int,
                    **int_spec,
                }
            )

        # Cardinality (ring XY-mixer method)
        if args.method in ("ring_xy", "both"):
            ring_xy_solver = build_ring_xy(
                experiment, data, selection_weighting=args.selection_weighting
            )
            N = ring_xy_solver.num_assets
            if args.sweep:
                K_list = list(range(args.k_min, N + 1))
            else:
                K = resolve_best_K(
                    ring_xy_solver, pid, results, args.rederive_k, k_min=args.k_min
                )
                if K is None:
                    print(
                        f"  no best_K resolvable; skipping cardinality for problem {pid}"
                    )
                    continue
                K_list = [K]
            for K in K_list:
                qcirc, _, layers_K, n_q_K = ring_xy_solver.build_cardinality_circuit(
                    K, device_name="default.qubit"
                )
                dummy_K = np.zeros(2 * layers_K)
                try:
                    K_spec = collect_specs(qcirc, dummy_K)
                except Exception as e:
                    print(f"  cardinality K={K} specs failed: {e}")
                    K_spec = {"error": str(e)}
                rows.append(
                    {
                        "problem_id": pid,
                        "n_assets": N,
                        "method": "cardinality",
                        "K": K,
                        "n_qubits": n_q_K,
                        "layers": layers_K,
                        "num_params": 2 * layers_K,
                        **K_spec,
                    }
                )
    return {"static_specs": rows}


def _hopo_live_row(pid, portfolio_hubo, n_qubits):
    """Recompute an integer-HUBO convergence row live (the default path and the
    --hopo-source json fallback). Returns the row dict, or an error stub."""
    E_min, E_max = None, None
    if n_qubits is not None and n_qubits <= 13:
        try:
            solve = portfolio_hubo.solve_exactly()
            eigenvalues = solve[5]
            E_min = float(min(eigenvalues))
            E_max = float(max(eigenvalues))
        except Exception as e:
            print(f"  solve_exactly failed: {e}")

    t0 = time.perf_counter()
    try:
        (_, fin_exp_int, _, _, _, _, hist_int, obj_vals_int, result_wb_int) = (
            portfolio_hubo.solve_with_qaoa_cma_es()
        )
        wall_int = time.perf_counter() - t0
        ar_int = None
        if E_min is not None and E_max is not None and E_max > E_min:
            # min-max ratio, 0 = optimal; profile_visualize plots 1 - this
            ar_int = float((float(fin_exp_int) - E_min) / (E_max - E_min))
        if ar_int is not None and not (-0.01 <= ar_int <= 1.01):
            print(
                f"  problem {pid}: approximation ratio {ar_int:.4g} outside "
                f"[0, 1]; dropping as corrupt"
            )
            ar_int = None
        budget_util, over_budget = _hopo_budget_util(
            obj_vals_int, result_wb_int, getattr(portfolio_hubo, "budget", None)
        )
        return {
            "problem_id": pid,
            "method": "integer_hubo",
            "K": None,
            "n_qubits": n_qubits,
            "n_assets": portfolio_hubo.num_assets,
            "source": "live",
            "wall_clock_seconds": float(wall_int),
            "evaluations": int(hist_int.get("evaluations", 0)),
            "iterations": int(hist_int.get("iterations", 0)),
            "final_expectation_value": float(fin_exp_int),
            "post_objective": float(max(obj_vals_int)) if obj_vals_int else None,
            "approximation_ratio": ar_int,
            "E_min": E_min,
            "E_max": E_max,
            "spectrum_partial": False,
            "budget_util": budget_util,
            "over_budget": over_budget,
            "training_history": hist_int,
        }
    except Exception as e:
        print(f"  integer-HUBO solve failed: {e}")
        return {"problem_id": pid, "method": "integer_hubo", "error": str(e)}


def run_convergence(args, problems, results):
    rows = []
    per_qubit_filled = {}
    want_hopo = args.method in ("hopo", "both")
    want_ring_xy = args.method in ("ring_xy", "both")
    cache = PriceCache()  # per-run price cache; each ticker downloaded at most once
    for pid, experiment in enumerate(problems):
        if args.id_range is not None and not (
            args.id_range[0] <= pid <= args.id_range[1]
        ):
            continue
        # When --hopo-source json supplies a covering entry, the integer-HUBO row
        # is read from disk; pull n_qubits from the entry so a pure hopo+json run
        # skips both the yfinance download and the HOPO build below.
        hopo_json = want_hopo and getattr(args, "hopo_source", "live") == "json"
        hopo_entry = None
        if hopo_json:
            hopo_entry = (getattr(args, "hopo_results", None) or {}).get(str(pid))

        # Likewise --card-source json reads the cardinality row from the merged
        # ring_xy results stream (the same dict resolve_best_K reads best_K from).
        # --sweep/--rederive-k explicitly request recomputation, so they bypass it.
        card_json = (
            want_ring_xy
            and getattr(args, "card_source", "live") == "json"
            and not args.sweep
            and not args.rederive_k
        )
        card_entry = (results or {}).get(str(pid)) if card_json else None

        data = None
        portfolio_hubo = None
        n_qubits = None
        if hopo_json and hopo_entry is not None:
            n_qubits = (hopo_entry.get("hyperparams") or {}).get("n_qubits")
        if n_qubits is None and not want_hopo and card_entry is not None:
            # ring_xy-only + card json: keep the live bucketing convention
            # (num_assets, not the integer-HUBO qubit count in hyperparams.n_qubits);
            # hyperparams.stocks is the post-ticker-drop asset list.
            n_qubits = (
                len((card_entry.get("hyperparams") or {}).get("stocks") or []) or None
            )
        if n_qubits is None:
            try:
                data = prepare_problem_data(experiment, cache)
            except Exception as e:
                print(f"  build failed: {e}")
                continue
            if want_hopo:
                portfolio_hubo = build_portfolio_hubo(experiment, data)
                n_qubits = portfolio_hubo.get_n_qubits()
            else:
                # ring_xy-only: bucket by the cardinality circuit's qubit count (=num_assets).
                n_qubits = len(data["expected_returns"])

        if per_qubit_filled.get(n_qubits, 0) >= args.per_qubit:
            continue
        per_qubit_filled[n_qubits] = per_qubit_filled.get(n_qubits, 0) + 1
        print(f"[convergence] problem {pid} n_qubits={n_qubits}")

        # Integer-HUBO (original paper method)
        if want_hopo:
            row = None
            if hopo_json:
                if hopo_entry is None:
                    print(
                        f"  no HOPO results entry for problem {pid}; recomputing live"
                    )
                else:
                    row = hopo_row_from_json(pid, hopo_entry)
                    if row is None:
                        print(
                            f"  HOPO entry for problem {pid} unusable; recomputing live"
                        )
            if row is not None:
                rows.append(row)
            else:
                if portfolio_hubo is None:
                    portfolio_hubo = build_portfolio_hubo(experiment, data)
                rows.append(_hopo_live_row(pid, portfolio_hubo, n_qubits))

        # Cardinality (ring XY-mixer method)
        if want_ring_xy:
            card_row = None
            if card_json:
                if card_entry is None:
                    print(
                        f"  no ring_xy results entry for problem {pid}; recomputing live"
                    )
                elif _recorded_no_feasible_K(card_entry):
                    # An explicit best_K=null is an authoritative "no feasible K"
                    # record, not a gap in the file — skip without the live
                    # fallback (which would download data and re-derive only to
                    # skip anyway).
                    print(
                        f"  ring_xy entry for problem {pid} records best_K=null "
                        f"(no feasible K); skipping cardinality"
                    )
                    continue
                else:
                    card_row = card_row_from_json(pid, card_entry, n_qubits)
                    if card_row is None:
                        print(
                            f"  ring_xy entry for problem {pid} unusable; recomputing live"
                        )
            if card_row is not None:
                rows.append(card_row)
                continue
            if data is None:
                # the hopo-json/card-json path skipped the shared download
                try:
                    data = prepare_problem_data(experiment, cache)
                except Exception as e:
                    print(f"  build failed: {e}")
                    continue
            ring_xy_solver = build_ring_xy(
                experiment, data, selection_weighting=args.selection_weighting
            )
            N = ring_xy_solver.num_assets
            if args.sweep:
                K_list = list(range(args.k_min, N + 1))
            else:
                K = resolve_best_K(
                    ring_xy_solver, pid, results, args.rederive_k, k_min=args.k_min
                )
                if K is None:
                    print(
                        f"  no best_K resolvable; skipping cardinality for problem {pid}"
                    )
                    continue
                K_list = [K]
            for K in K_list:
                t0 = time.perf_counter()
                try:
                    res = ring_xy_solver.solve_with_qaoa_cardinality(K)
                    wall_K = time.perf_counter() - t0
                    # Recompute the global min-max ratio in profile.py from the
                    # exact bounds in `res` (same cached selection diagonal the
                    # reference uses) rather than trusting res["approximation_
                    # ratio_global"] — symmetric to the integer-HUBO live row,
                    # which recomputes from the exact spectrum it already holds.
                    ar_global = _card_global_ar(
                        res["selection_energy_global_min"],
                        res["selection_energy_global_max"],
                        float(res["final_expectation_value"]),
                        pid,
                    )
                    budget_util, over_budget = _budget_util(
                        None if res.get("infeasible") else res.get("realized_budget"),
                        getattr(ring_xy_solver, "budget", None),
                    )
                    rows.append(
                        {
                            "problem_id": pid,
                            "method": "cardinality",
                            "K": K,
                            # The cardinality circuit uses one qubit per asset, so its
                            # qubit count is N (num_assets) — NOT the shared `n_qubits`,
                            # which in --method both is the integer-HUBO log-encoding
                            # count and can exceed N. Bucketing still keys on `n_qubits`.
                            "n_qubits": N,
                            "n_assets": N,
                            "source": "live",
                            "wall_clock_seconds": float(wall_K),
                            "evaluations": int(
                                res["training_history"].get("evaluations", 0)
                            ),
                            "iterations": int(res["iterations"]),
                            "final_expectation_value": float(
                                res["final_expectation_value"]
                            ),
                            "approximation_ratio_subspace": res[
                                "approximation_ratio_subspace"
                            ],
                            "approximation_ratio_global": ar_global,
                            "approximation_ratio_global_source": "reference",
                            "E_subspace_min": res["selection_energy_min_subspace"],
                            "E_subspace_max": res["selection_energy_max_subspace"],
                            "E_dicke_mean": res["selection_energy_dicke_mean"],
                            "E_global_min": res["selection_energy_global_min"],
                            "E_global_max": res["selection_energy_global_max"],
                            "post_objective": res["post_objective"],
                            "infeasible": res["infeasible"],
                            "budget_util": budget_util,
                            "over_budget": over_budget,
                            # The freshly-solved portfolio's allocation, so the
                            # benefit-vs-N profit term reflects the SAME stochastic
                            # solve as post_objective (Gap-1 coherence), not the
                            # (different) saved best_K allocation. CMA-ES is unseeded.
                            "allocation": res.get("allocation"),
                            "training_history": res["training_history"],
                        }
                    )
                except Exception as e:
                    print(f"  cardinality K={K} solve failed: {e}")
                    rows.append(
                        {
                            "problem_id": pid,
                            "method": "cardinality",
                            "K": K,
                            "error": str(e),
                        }
                    )

    # Per-problem continuous-baseline + exact-reference enrichment for the Fig. 6
    # comparison (profile_visualize --convergence). Applied as a post-pass so every
    # row — including the ones reached via the cardinality `continue` branches — is
    # enriched. The continuous baseline is shared across both method streams; pull
    # it (and, for ring_xy files, selection_exact) from the merged `results` stream,
    # falling back to the HOPO stream. Available even in fully-live runs because
    # `results` is loaded unconditionally in main().
    #
    # The benefit-vs-N profit term additionally needs price-weighted expected
    # returns (μ + prices), which are NOT saved: build per-problem ticker maps by
    # recomputing from the entry's hyperparams through the shared per-run `cache`
    # (a live run already cached these series, so it is a cache hit and the μ
    # matches the live solve exactly). `method_return` is computed from each row's
    # OWN allocation (live = fresh solve, json = saved best_K) for Gap-1 coherence
    # with that row's post_objective; `continuous_return` comes from the entry's
    # LP-deterministic continuous allocation.
    variant = getattr(args, "continuous_variant", "constrained")
    hopo_results = getattr(args, "hopo_results", None) or {}
    enrich_cache = {}
    maps_cache = {}
    for row in rows:
        pid = row.get("problem_id")
        if pid is None:
            continue
        entry = (results or {}).get(str(pid)) or hopo_results.get(str(pid))
        if pid not in maps_cache:
            hp = (entry or {}).get("hyperparams") or {}
            maps_cache[pid] = _recompute_return_maps(hp, cache)
        mu_map, price_map = maps_cache[pid]
        if pid not in enrich_cache:
            enrich_cache[pid] = compare_enrichment(entry, variant, mu_map, price_map)
        row.update(enrich_cache[pid])
        # Per-row method profit from this row's own portfolio (Gap-1 coherence).
        alloc = row.get("allocation")
        row["method_return"] = (
            _expected_return_dollars(alloc, mu_map, price_map)
            if (alloc is not None and mu_map and price_map)
            else None
        )
    return {"convergence": rows}


def main():
    parser = argparse.ArgumentParser(description="Profile QAOA pipelines")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    def add_common(p):
        p.add_argument(
            "start",
            type=int,
            nargs="?",
            default=None,
            help="First experiment id to profile (0-based, inclusive). "
            "Omit to profile all problems.",
        )
        p.add_argument(
            "end",
            type=int,
            nargs="?",
            default=None,
            help="Last experiment id to profile (0-based, inclusive). "
            "Defaults to start when only start is given.",
        )
        p.add_argument("--experiments-json", default="experiments_data.json")
        p.add_argument(
            "--method",
            choices=["hopo", "ring_xy", "both"],
            default="both",
            help="Which method(s) to profile: 'hopo' (integer-HUBO rows), "
            "'ring_xy' (cardinality rows), or 'both' (default).",
        )
        p.add_argument("--results-json", default=None)
        p.add_argument(
            "--hopo-source",
            choices=["live", "json"],
            default="live",
            help="convergence only: 'live' (default) recomputes the "
            "integer-HUBO baseline with CMA-ES; 'json' loads it "
            "from saved results/portfolio_optimization_exp_*.json "
            "(falls back to live per-experiment when absent).",
        )
        p.add_argument(
            "--hopo-results-json",
            default=None,
            help="Explicit current-format HOPO results file for "
            "--hopo-source json (default: merge from results/).",
        )
        p.add_argument(
            "--card-source",
            choices=["live", "json"],
            default="live",
            help="convergence only: 'live' (default) re-runs the "
            "fixed-K cardinality QAOA; 'json' loads the best_K "
            "per_K entry from the ring_xy results JSON (the same "
            "merged stream used for best_K resolution; falls "
            "back to live per-experiment when absent).",
        )
        p.add_argument(
            "--selection-weighting",
            choices=["budget", "unit"],
            default="budget",
            help="ring_xy live runs only (--card-source live / --rederive-k / "
            "--sweep): selection HUBO weighting, must match the saved results "
            "you compare against (TASK-03). 'budget' (default) = share-count "
            "weighted; 'unit' = original price-blind score. Has no effect on the "
            "--card-source json path, which reads weighting-agnostic post_objective.",
        )
        p.add_argument(
            "--rederive-k",
            action="store_true",
            help="Run K sweep to find best_K instead of reading from results JSON.",
        )
        p.add_argument(
            "--sweep",
            action="store_true",
            help="Profile every K from k_min..N instead of just best-K.",
        )
        p.add_argument(
            "--k-min",
            type=int,
            default=1,
            help="Smallest cardinality K for --sweep/--rederive-k "
            "(default 1; K=0 is excluded because the XY mixer "
            "freezes the empty-selection state).",
        )
        p.add_argument(
            "--out",
            default=None,
            help="Output JSON path (default: "
            "profile_results_<subcommand>_<timestamp>.json).",
        )
        p.add_argument(
            "--max-problems",
            type=int,
            default=None,
            help="Limit number of problems processed (in order).",
        )

    p_specs = sub.add_parser("specs", help="Static circuit specs (no CMA-ES).")
    add_common(p_specs)

    p_conv = sub.add_parser(
        "convergence", help="Full CMA-ES + wall-clock + approx ratio."
    )
    add_common(p_conv)
    p_conv.add_argument(
        "--per-qubit",
        type=int,
        default=1,
        help="Problems per qubit count for convergence (default 1).",
    )
    p_conv.add_argument(
        "--continuous-variant",
        choices=["constrained", "unconstrained", "both"],
        default="constrained",
        help="convergence only: which classical continuous "
        "baseline is the Fig. 6 reference series. "
        "'constrained' (default) = budget-constrained solve + "
        "LP rounding (always stored, matches paper Fig. 6); "
        "'unconstrained' = penalty variant (Eq. 19, None "
        "without higher moments); 'both' keeps constrained "
        "primary and emits the unconstrained series too.",
    )

    args = parser.parse_args()

    # Default output name encodes which subcommand produced it; --out overrides.
    if args.out is None:
        args.out = f"results_profile/profile_results_{args.subcommand}_{timestamp}.json"

    # Resolve the inclusive experiment range (0-based). None => all problems.
    if args.start is None:
        args.id_range = None
    else:
        end = args.end if args.end is not None else args.start
        if end < args.start:
            print(
                f"Error: invalid range start={args.start} end={end} (end must be >= start)"
            )
            sys.exit(1)
        args.id_range = (args.start, end)

    problems = load_problems(args.experiments_json)
    if args.max_problems is not None:
        problems = problems[: args.max_problems]

    # Assemble the results dict that resolve_best_K reads from. An explicit
    # --results-json takes one file as-is; otherwise pull each requested
    # experiment from the newest file in results/ that contains it.
    provenance = {}
    if args.results_json is not None:
        results = load_results(args.results_json)
        results_files = [args.results_json] if results is not None else []
        if results is not None:
            provenance = {k: args.results_json for k in results}
    else:
        results, provenance, results_files = build_merged_results(
            args.method, args.id_range
        )
        if not results:
            results = None
    if results_files:
        print("Results files used (newest-first):")
        for path in results_files:
            print(f"  {path}")

    # The ring_xy results JSON feeds best_K resolution and (for convergence
    # --card-source json) the full cardinality rows.
    if args.method in ("ring_xy", "both") and results is None and not args.rederive_k:
        print("Warning: no ring_xy results JSON found; re-deriving best_K everywhere.")
        args.rederive_k = True

    # --card-source json interaction notes (convergence cardinality rows only).
    if (
        args.subcommand == "convergence"
        and args.method in ("ring_xy", "both")
        and args.card_source == "json"
    ):
        if results is None:
            print(
                "Warning: --card-source json but no ring_xy results JSON found; "
                "recomputing cardinality live everywhere."
            )
        elif args.sweep or args.rederive_k:
            print(
                "Note: --card-source json is bypassed by --sweep/--rederive-k; "
                "recomputing cardinality live."
            )

    # HOPO-with-QAOA JSON stream (independent of the ring_xy best_K stream) for
    # convergence --hopo-source json. Only assembled when actually requested.
    args.hopo_results = None
    hopo_results_files = []
    hopo_provenance = {}
    if (
        args.subcommand == "convergence"
        and args.method in ("hopo", "both")
        and getattr(args, "hopo_source", "live") == "json"
    ):
        args.hopo_results, hopo_provenance, hopo_results_files = build_hopo_results(
            args.id_range, args.hopo_results_json
        )
        if hopo_results_files:
            print("HOPO results files used (newest-first):")
            for path in hopo_results_files:
                print(f"  {path}")
        if not args.hopo_results:
            print(
                "Warning: --hopo-source json but no HOPO results JSON found; "
                "recomputing integer-HUBO live everywhere."
            )

    meta = {
        "subcommand": args.subcommand,
        "method": args.method,
        "experiments_json": args.experiments_json,
        "requested_range": list(args.id_range) if args.id_range is not None else None,
        "results_files": results_files,
        "results_provenance": provenance,
        "rederive_k": args.rederive_k,
        "sweep": args.sweep,
        "k_min": args.k_min,
        "max_problems": args.max_problems,
    }
    if args.subcommand == "convergence":
        meta["per_qubit"] = args.per_qubit
        meta["hopo_source"] = args.hopo_source
        meta["hopo_results_files"] = hopo_results_files
        meta["hopo_results_provenance"] = hopo_provenance
        meta["card_source"] = args.card_source
        meta["continuous_variant"] = args.continuous_variant
        body = run_convergence(args, problems, results)
    else:
        body = run_specs(args, problems, results)

    output = {"meta": meta, **body}
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
