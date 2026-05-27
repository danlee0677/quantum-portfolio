"""Profile / benchmarking tool for the HUBO and cardinality QAOA pipelines.

Subcommands:
  specs        — static circuit metrics (qubits, depth, gate counts) on all 100
                 problems; cheap.
  convergence  — CMA-ES dynamics (evaluations, wall-clock, approx ratio) on a
                 small subset; expensive.

Best-K resolution (shared):
  default → read cardinality_qaoa_solution.best_K from the latest results JSON.
  --rederive-k or no results JSON found → run solve_with_qaoa_cardinality on
  every K to find the best.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pennylane as qml
import yfinance as yf
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov

from coskweness_cokurtosis import coskewness, cokurtosis
from portfolio_hubo_qaoa_light import HigherOrderPortfolioQAOA
from utils import replace_h_rz_h_with_rx


HYPERPARAMS = {
    "risk_aversion": 0.1,
    "max_qubits": 15,
    "log_encoding": True,
    "strict_budget_constraint": False,
    "lambda_budget": 0.001,
}

ALLOWED_GATES = ["CNOT", "RZ", "RX", "Hadamard"]


def load_problems(path):
    with open(path, "r") as f:
        return list(json.load(f)["data"])


def discover_results_json():
    candidates = sorted(glob.glob("portfolio_optimization_batch_*.json"))
    return candidates[-1] if candidates else None


def load_results(path):
    if path is None or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def build_portfolio_hubo(experiment):
    stocks = experiment["stocks"]
    start = experiment["start"]
    end = experiment["end"]
    budget = experiment["budget"]
    data = yf.download(stocks, start=start, end=end, progress=False, auto_adjust=True)
    prices_now = data["Close"].iloc[-1]
    returns = data["Close"].pct_change(fill_method=None).dropna(how="any")
    stocks = returns.columns
    numpy_returns = returns.to_numpy()
    expected_returns = mean_historical_return(returns, returns_data=True, compounding=False).to_numpy()
    covariance_matrix = sample_cov(returns, returns_data=True).to_numpy()
    coskewness_tensor = coskewness(numpy_returns)
    cokurtosis_tensor = cokurtosis(numpy_returns)
    return HigherOrderPortfolioQAOA(
        stocks=stocks,
        prices_now=prices_now,
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        budget=budget,
        coskewness_tensor=coskewness_tensor,
        cokurtosis_tensor=cokurtosis_tensor,
        **HYPERPARAMS,
    )


def _rederive_best_K(portfolio_hubo):
    N = portfolio_hubo.num_assets
    best_obj = None
    best_K = None
    for K in range(2, N + 1):
        try:
            res = portfolio_hubo.solve_with_qaoa_cardinality(K)
        except Exception as e:
            print(f"  re-derive K={K} failed: {e}")
            continue
        if res["infeasible"] or res["post_objective"] is None:
            continue
        if best_obj is None or res["post_objective"] > best_obj:
            best_obj = res["post_objective"]
            best_K = K
    return best_K


def resolve_best_K(portfolio_hubo, problem_id, results, rederive):
    if rederive or results is None:
        return _rederive_best_K(portfolio_hubo)
    entry = results.get(str(problem_id))
    if entry is None:
        print(f"  results JSON has no entry for problem {problem_id}; re-deriving best_K")
        return _rederive_best_K(portfolio_hubo)
    card = entry.get("cardinality_qaoa_solution") or {}
    best_K = card.get("best_K")
    if best_K is None:
        print(f"  results JSON missing best_K for problem {problem_id}; re-deriving")
        return _rederive_best_K(portfolio_hubo)
    return int(best_K)


def compile_for_specs(circuit):
    dispatched = qml.transform(replace_h_rz_h_with_rx)
    compiled = qml.compile(circuit, basis_set=ALLOWED_GATES)
    compiled = qml.compile(compiled, pipeline=[dispatched])
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
    for pid, experiment in enumerate(problems):
        print(f"[specs] problem {pid} stocks={len(experiment['stocks'])}")
        try:
            portfolio_hubo = build_portfolio_hubo(experiment)
        except Exception as e:
            print(f"  build failed: {e}")
            rows.append({"problem_id": pid, "error": f"build: {e}"})
            continue
        N = portfolio_hubo.num_assets
        n_qubits = portfolio_hubo.get_n_qubits()

        # Integer-HUBO
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

        # Cardinality
        if args.sweep:
            K_list = list(range(2, N + 1))
        else:
            K = resolve_best_K(portfolio_hubo, pid, results, args.rederive_k)
            if K is None:
                print(f"  no best_K resolvable; skipping cardinality for problem {pid}")
                continue
            K_list = [K]
        for K in K_list:
            qcirc, _, layers_K, n_q_K = portfolio_hubo.build_cardinality_circuit(K, device_name="default.qubit")
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


def run_convergence(args, problems, results):
    rows = []
    per_qubit_filled = {}
    for pid, experiment in enumerate(problems):
        try:
            portfolio_hubo = build_portfolio_hubo(experiment)
        except Exception as e:
            print(f"  build failed: {e}")
            continue
        n_qubits = portfolio_hubo.get_n_qubits()
        if per_qubit_filled.get(n_qubits, 0) >= args.per_qubit:
            continue
        per_qubit_filled[n_qubits] = per_qubit_filled.get(n_qubits, 0) + 1
        print(f"[convergence] problem {pid} n_qubits={n_qubits}")

        E_min, E_max = None, None
        if n_qubits <= 13:
            try:
                solve = portfolio_hubo.solve_exactly()
                eigenvalues = solve[5]
                E_min = float(min(eigenvalues))
                E_max = float(max(eigenvalues))
            except Exception as e:
                print(f"  solve_exactly failed: {e}")

        # Integer-HUBO
        t0 = time.perf_counter()
        try:
            (_, fin_exp_int, _, _, _, _, hist_int, _, _) = portfolio_hubo.solve_with_qaoa_cma_es()
            wall_int = time.perf_counter() - t0
            ar_int = None
            if E_min is not None and E_max is not None and E_max > E_min:
                ar_int = float((float(fin_exp_int) - E_min) / (E_max - E_min))
            rows.append({
                "problem_id": pid, "method": "integer_hubo", "K": None,
                "n_qubits": n_qubits,
                "wall_clock_seconds": float(wall_int),
                "evaluations": int(hist_int.get("evaluations", 0)),
                "iterations": int(hist_int.get("iterations", 0)),
                "final_expectation_value": float(fin_exp_int),
                "approximation_ratio": ar_int,
                "E_min": E_min, "E_max": E_max,
                "training_history": hist_int,
            })
        except Exception as e:
            print(f"  integer-HUBO solve failed: {e}")
            rows.append({"problem_id": pid, "method": "integer_hubo", "error": str(e)})

        # Cardinality K choice
        N = portfolio_hubo.num_assets
        if args.sweep:
            K_list = list(range(2, N + 1))
        else:
            K = resolve_best_K(portfolio_hubo, pid, results, args.rederive_k)
            if K is None:
                print(f"  no best_K resolvable; skipping cardinality for problem {pid}")
                continue
            K_list = [K]
        for K in K_list:
            t0 = time.perf_counter()
            try:
                res = portfolio_hubo.solve_with_qaoa_cardinality(K)
                wall_K = time.perf_counter() - t0
                rows.append({
                    "problem_id": pid, "method": "cardinality", "K": K,
                    "n_qubits": n_qubits,
                    "wall_clock_seconds": float(wall_K),
                    "evaluations": int(res["training_history"].get("evaluations", 0)),
                    "iterations": int(res["iterations"]),
                    "final_expectation_value": float(res["final_expectation_value"]),
                    "approximation_ratio": None,
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
        p.add_argument("--experiments-json", default="experiments_data.json")
        p.add_argument("--results-json", default=None)
        p.add_argument("--rederive-k", action="store_true",
                       help="Run K sweep to find best_K instead of reading from results JSON.")
        p.add_argument("--sweep", action="store_true",
                       help="Profile every K from 2..N instead of just best-K.")
        p.add_argument("--out", default="profile_results.json")
        p.add_argument("--max-problems", type=int, default=None,
                       help="Limit number of problems processed (in order).")

    p_specs = sub.add_parser("specs", help="Static circuit specs (no CMA-ES).")
    add_common(p_specs)

    p_conv = sub.add_parser("convergence", help="Full CMA-ES + wall-clock + approx ratio.")
    add_common(p_conv)
    p_conv.add_argument("--per-qubit", type=int, default=1,
                        help="Problems per qubit count for convergence (default 1).")

    args = parser.parse_args()

    problems = load_problems(args.experiments_json)
    if args.max_problems is not None:
        problems = problems[: args.max_problems]
    results_path = args.results_json or discover_results_json()
    results = load_results(results_path)
    if results is None and not args.rederive_k:
        print("Warning: no results JSON found; re-deriving best_K everywhere.")
        args.rederive_k = True

    meta = {
        "subcommand": args.subcommand,
        "experiments_json": args.experiments_json,
        "results_json": results_path,
        "rederive_k": args.rederive_k,
        "sweep": args.sweep,
        "max_problems": args.max_problems,
    }
    if args.subcommand == "convergence":
        meta["per_qubit"] = args.per_qubit
        body = run_convergence(args, problems, results)
    else:
        body = run_specs(args, problems, results)

    output = {"meta": meta, **body}
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
