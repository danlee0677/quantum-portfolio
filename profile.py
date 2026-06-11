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

import numpy as np
import pennylane as qml
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov

from data_provider import fetch_close_prices, PriceCache
from coskweness_cokurtosis import coskewness, cokurtosis
from portfolio_hubo_qaoa_light import HigherOrderPortfolioQAOA
from ring_xy import RingXYCardinalityQAOA
from utils import replace_h_rz_h_with_rx, decompose_ry
from datetime import datetime


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
        close = fetch_close_prices(stocks, start=start, end=end,
                                   auto_adjust=True, progress=False).close
    prices_now = close.iloc[-1]
    returns = close.pct_change(fill_method=None).dropna(how="any")
    stocks = returns.columns
    numpy_returns = returns.to_numpy()
    expected_returns = mean_historical_return(returns, returns_data=True, compounding=False).to_numpy()
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


def build_ring_xy(experiment, data=None):
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
    )


def _rederive_best_K(ring_xy_solver):
    N = ring_xy_solver.num_assets
    best_obj = None
    best_K = None
    for K in range(2, N + 1):
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


def resolve_best_K(ring_xy_solver, problem_id, results, rederive):
    if rederive or results is None:
        return _rederive_best_K(ring_xy_solver)
    entry = results.get(str(problem_id))
    if entry is None:
        print(f"  results JSON has no entry for problem {problem_id}; re-deriving best_K")
        return _rederive_best_K(ring_xy_solver)
    card = entry.get("cardinality_qaoa_solution") or {}
    best_K = card.get("best_K")
    if best_K is None:
        print(f"  results JSON missing best_K for problem {problem_id}; re-deriving")
        return _rederive_best_K(ring_xy_solver)
    return int(best_K)


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
    if not isinstance(qaoa, dict) or "error" in entry or "final_expectation_value" not in qaoa:
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
            print(f"  problem {pid}: partial exact spectrum ({len(spectrum)} of "
                  f"2^{n_qubits} states); approximation ratio unavailable")
    else:
        eigs = exact.get("smallest_eigenvalues")
        if eigs:
            E_min = float(eigs[0])

    ar = None
    if E_min is not None and E_max is not None and E_max > E_min:
        # min-max ratio, 0 = optimal; profile_visualize plots 1 - this
        ar = float((fin_exp - E_min) / (E_max - E_min))
    if ar is not None and not (-0.01 <= ar <= 1.01):
        print(f"  problem {pid}: approximation ratio {ar:.4g} outside [0, 1]; "
              f"dropping as corrupt")
        ar = None

    return {
        "problem_id": pid, "method": "integer_hubo", "K": None,
        "n_qubits": int(n_qubits) if n_qubits is not None else None,
        "n_assets": n_assets,
        "source": "json",
        "wall_clock_seconds": None,
        "evaluations": int(hist.get("evaluations", 0)),
        "iterations": int(hist.get("iterations", 0)),
        "final_expectation_value": fin_exp,
        "post_objective": post_objective,
        "approximation_ratio": ar,
        "E_min": E_min, "E_max": E_max,
        "spectrum_partial": spectrum_partial,
        "training_history": hist,
    }


def card_row_from_json(pid, entry, n_qubits):
    """Build a cardinality convergence row from a saved ring_xy entry instead of
    re-running the fixed-K QAOA. Returns None when the entry lacks a usable
    best_K per_K entry so the caller can fall back to a live solve.

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
    e = next((p for p in (card.get("per_K") or [])
              if isinstance(p, dict) and p.get("K") == best_K
              and "error" not in p and "final_expectation_value" in p), None)
    if e is None:
        return None

    num_assets = len((entry.get("hyperparams") or {}).get("stocks") or []) or n_qubits

    return {
        "problem_id": pid, "method": "cardinality", "K": int(best_K),
        "n_qubits": num_assets,
        "n_assets": num_assets,
        "source": "json",
        "wall_clock_seconds": None,
        "evaluations": int((e.get("training_history") or {}).get("evaluations", 0)),
        "iterations": int(e.get("iterations", 0)),
        "final_expectation_value": float(e["final_expectation_value"]),
        "approximation_ratio_subspace": e.get("approximation_ratio_subspace"),
        "approximation_ratio_global": e.get("approximation_ratio_global"),
        "E_subspace_min": e.get("selection_energy_min_subspace"),
        "E_subspace_max": e.get("selection_energy_max_subspace"),
        "E_dicke_mean": e.get("selection_energy_dicke_mean"),
        "E_global_min": e.get("selection_energy_global_min"),
        "E_global_max": e.get("selection_energy_global_max"),
        "post_objective": e.get("post_objective"),
        "infeasible": e.get("infeasible"),
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
    unexpected = sorted(set(gate_types) - set(ALLOWED_GATES) - {"Identity", "GlobalPhase"})
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
        if args.id_range is not None and not (args.id_range[0] <= pid <= args.id_range[1]):
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
            qcirc, _, layers_int, n_q_int = portfolio_hubo.build_integer_hubo_circuit(device_name="default.qubit")
            dummy_int = np.zeros(2 * layers_int)
            try:
                int_spec = collect_specs(qcirc, dummy_int)
            except Exception as e:
                print(f"  integer-HUBO specs failed: {e}")
                int_spec = {"error": str(e)}
            rows.append({
                "problem_id": pid, "n_assets": N, "method": "integer_hubo",
                "K": None, "n_qubits": n_q_int, "layers": layers_int,
                "num_params": 2 * layers_int, **int_spec,
            })

        # Cardinality (ring XY-mixer method)
        if args.method in ("ring_xy", "both"):
            ring_xy_solver = build_ring_xy(experiment, data)
            N = ring_xy_solver.num_assets
            if args.sweep:
                K_list = list(range(2, N + 1))
            else:
                K = resolve_best_K(ring_xy_solver, pid, results, args.rederive_k)
                if K is None:
                    print(f"  no best_K resolvable; skipping cardinality for problem {pid}")
                    continue
                K_list = [K]
            for K in K_list:
                qcirc, _, layers_K, n_q_K = ring_xy_solver.build_cardinality_circuit(K, device_name="default.qubit")
                dummy_K = np.zeros(2 * layers_K)
                try:
                    K_spec = collect_specs(qcirc, dummy_K)
                except Exception as e:
                    print(f"  cardinality K={K} specs failed: {e}")
                    K_spec = {"error": str(e)}
                rows.append({
                    "problem_id": pid, "n_assets": N, "method": "cardinality",
                    "K": K, "n_qubits": n_q_K, "layers": layers_K,
                    "num_params": 2 * layers_K, **K_spec,
                })
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
        (_, fin_exp_int, _, _, _, _, hist_int, obj_vals_int, _) = portfolio_hubo.solve_with_qaoa_cma_es()
        wall_int = time.perf_counter() - t0
        ar_int = None
        if E_min is not None and E_max is not None and E_max > E_min:
            # min-max ratio, 0 = optimal; profile_visualize plots 1 - this
            ar_int = float((float(fin_exp_int) - E_min) / (E_max - E_min))
        if ar_int is not None and not (-0.01 <= ar_int <= 1.01):
            print(f"  problem {pid}: approximation ratio {ar_int:.4g} outside "
                  f"[0, 1]; dropping as corrupt")
            ar_int = None
        return {
            "problem_id": pid, "method": "integer_hubo", "K": None,
            "n_qubits": n_qubits,
            "n_assets": portfolio_hubo.num_assets,
            "source": "live",
            "wall_clock_seconds": float(wall_int),
            "evaluations": int(hist_int.get("evaluations", 0)),
            "iterations": int(hist_int.get("iterations", 0)),
            "final_expectation_value": float(fin_exp_int),
            "post_objective": float(max(obj_vals_int)) if obj_vals_int else None,
            "approximation_ratio": ar_int,
            "E_min": E_min, "E_max": E_max,
            "spectrum_partial": False,
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
        if args.id_range is not None and not (args.id_range[0] <= pid <= args.id_range[1]):
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
        card_json = (want_ring_xy and getattr(args, "card_source", "live") == "json"
                     and not args.sweep and not args.rederive_k)
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
            n_qubits = len((card_entry.get("hyperparams") or {}).get("stocks") or []) or None
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
                    print(f"  no HOPO results entry for problem {pid}; recomputing live")
                else:
                    row = hopo_row_from_json(pid, hopo_entry)
                    if row is None:
                        print(f"  HOPO entry for problem {pid} unusable; recomputing live")
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
                    print(f"  no ring_xy results entry for problem {pid}; recomputing live")
                else:
                    card_row = card_row_from_json(pid, card_entry, n_qubits)
                    if card_row is None:
                        print(f"  ring_xy entry for problem {pid} unusable; recomputing live")
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
            ring_xy_solver = build_ring_xy(experiment, data)
            N = ring_xy_solver.num_assets
            if args.sweep:
                K_list = list(range(2, N + 1))
            else:
                K = resolve_best_K(ring_xy_solver, pid, results, args.rederive_k)
                if K is None:
                    print(f"  no best_K resolvable; skipping cardinality for problem {pid}")
                    continue
                K_list = [K]
            for K in K_list:
                t0 = time.perf_counter()
                try:
                    res = ring_xy_solver.solve_with_qaoa_cardinality(K)
                    wall_K = time.perf_counter() - t0
                    rows.append({
                        "problem_id": pid, "method": "cardinality", "K": K,
                        # The cardinality circuit uses one qubit per asset, so its
                        # qubit count is N (num_assets) — NOT the shared `n_qubits`,
                        # which in --method both is the integer-HUBO log-encoding
                        # count and can exceed N. Bucketing still keys on `n_qubits`.
                        "n_qubits": N,
                        "n_assets": N,
                        "source": "live",
                        "wall_clock_seconds": float(wall_K),
                        "evaluations": int(res["training_history"].get("evaluations", 0)),
                        "iterations": int(res["iterations"]),
                        "final_expectation_value": float(res["final_expectation_value"]),
                        "approximation_ratio_subspace": res["approximation_ratio_subspace"],
                        "approximation_ratio_global": res["approximation_ratio_global"],
                        "E_subspace_min": res["selection_energy_min_subspace"],
                        "E_subspace_max": res["selection_energy_max_subspace"],
                        "E_dicke_mean": res["selection_energy_dicke_mean"],
                        "E_global_min": res["selection_energy_global_min"],
                        "E_global_max": res["selection_energy_global_max"],
                        "post_objective": res["post_objective"],
                        "infeasible": res["infeasible"],
                        "training_history": res["training_history"],
                    })
                except Exception as e:
                    print(f"  cardinality K={K} solve failed: {e}")
                    rows.append({"problem_id": pid, "method": "cardinality", "K": K, "error": str(e)})
    return {"convergence": rows}


def main():
    parser = argparse.ArgumentParser(description="Profile QAOA pipelines")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    def add_common(p):
        p.add_argument("start", type=int, nargs="?", default=None,
                       help="First experiment id to profile (0-based, inclusive). "
                            "Omit to profile all problems.")
        p.add_argument("end", type=int, nargs="?", default=None,
                       help="Last experiment id to profile (0-based, inclusive). "
                            "Defaults to start when only start is given.")
        p.add_argument("--experiments-json", default="experiments_data.json")
        p.add_argument("--method", choices=["hopo", "ring_xy", "both"], default="both",
                       help="Which method(s) to profile: 'hopo' (integer-HUBO rows), "
                            "'ring_xy' (cardinality rows), or 'both' (default).")
        p.add_argument("--results-json", default=None)
        p.add_argument("--hopo-source", choices=["live", "json"], default="live",
                       help="convergence only: 'live' (default) recomputes the "
                            "integer-HUBO baseline with CMA-ES; 'json' loads it "
                            "from saved results/portfolio_optimization_exp_*.json "
                            "(falls back to live per-experiment when absent).")
        p.add_argument("--hopo-results-json", default=None,
                       help="Explicit current-format HOPO results file for "
                            "--hopo-source json (default: merge from results/).")
        p.add_argument("--card-source", choices=["live", "json"], default="live",
                       help="convergence only: 'live' (default) re-runs the "
                            "fixed-K cardinality QAOA; 'json' loads the best_K "
                            "per_K entry from the ring_xy results JSON (the same "
                            "merged stream used for best_K resolution; falls "
                            "back to live per-experiment when absent).")
        p.add_argument("--rederive-k", action="store_true",
                       help="Run K sweep to find best_K instead of reading from results JSON.")
        p.add_argument("--sweep", action="store_true",
                       help="Profile every K from 2..N instead of just best-K.")
        p.add_argument("--out", default=None,
                       help="Output JSON path (default: "
                            "profile_results_<subcommand>_<timestamp>.json).")
        p.add_argument("--max-problems", type=int, default=None,
                       help="Limit number of problems processed (in order).")

    p_specs = sub.add_parser("specs", help="Static circuit specs (no CMA-ES).")
    add_common(p_specs)

    p_conv = sub.add_parser("convergence", help="Full CMA-ES + wall-clock + approx ratio.")
    add_common(p_conv)
    p_conv.add_argument("--per-qubit", type=int, default=1,
                        help="Problems per qubit count for convergence (default 1).")

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
            print(f"Error: invalid range start={args.start} end={end} (end must be >= start)")
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
        results, provenance, results_files = build_merged_results(args.method, args.id_range)
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
    if args.subcommand == "convergence" and args.method in ("ring_xy", "both") \
            and args.card_source == "json":
        if results is None:
            print("Warning: --card-source json but no ring_xy results JSON found; "
                  "recomputing cardinality live everywhere.")
        elif args.sweep or args.rederive_k:
            print("Note: --card-source json is bypassed by --sweep/--rederive-k; "
                  "recomputing cardinality live.")

    # HOPO-with-QAOA JSON stream (independent of the ring_xy best_K stream) for
    # convergence --hopo-source json. Only assembled when actually requested.
    args.hopo_results = None
    hopo_results_files = []
    hopo_provenance = {}
    if (args.subcommand == "convergence"
            and args.method in ("hopo", "both")
            and getattr(args, "hopo_source", "live") == "json"):
        args.hopo_results, hopo_provenance, hopo_results_files = build_hopo_results(
            args.id_range, args.hopo_results_json)
        if hopo_results_files:
            print("HOPO results files used (newest-first):")
            for path in hopo_results_files:
                print(f"  {path}")
        if not args.hopo_results:
            print("Warning: --hopo-source json but no HOPO results JSON found; "
                  "recomputing integer-HUBO live everywhere.")

    meta = {
        "subcommand": args.subcommand,
        "method": args.method,
        "experiments_json": args.experiments_json,
        "requested_range": list(args.id_range) if args.id_range is not None else None,
        "results_files": results_files,
        "results_provenance": provenance,
        "rederive_k": args.rederive_k,
        "sweep": args.sweep,
        "max_problems": args.max_problems,
    }
    if args.subcommand == "convergence":
        meta["per_qubit"] = args.per_qubit
        meta["hopo_source"] = args.hopo_source
        meta["hopo_results_files"] = hopo_results_files
        meta["hopo_results_provenance"] = hopo_provenance
        meta["card_source"] = args.card_source
        body = run_convergence(args, problems, results)
    else:
        body = run_specs(args, problems, results)

    output = {"meta": meta, **body}
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
