import argparse
import json
import os
import sys
from datetime import datetime

from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov

from coskweness_cokurtosis import cokurtosis, coskewness
from data_provider import fetch_close_prices
from portfolio_hubo_qaoa_light import HigherOrderPortfolioQAOA
from ring_xy import RingXYCardinalityQAOA

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run a range of portfolio optimization experiments"
)
parser.add_argument(
    "start", type=int, help="First experiment id to process (0-based, inclusive)"
)
parser.add_argument(
    "end", type=int, help="Last experiment id to process (0-based, inclusive)"
)
parser.add_argument(
    "--method",
    choices=["hopo", "ring_xy"],
    default="hopo",
    help="Which QAOA algorithm to run: 'hopo' (paper integer-HUBO) or "
    "'ring_xy' (cardinality-selection). Shared classical/exact "
    "baselines are written either way.",
)
parser.add_argument(
    "--k-selector",
    choices=["lasso", "sweep", "both"],
    default="lasso",
    help="(ring_xy only) How to choose the cardinality K. 'lasso': "
    "cheap classical LASSO localizer + neighborhood scan, then "
    "QAOA on the top-k-hedge K (scales to large N). 'sweep': "
    "brute-force QAOA over every K in [k_min, N] (original "
    "behavior). 'both': run the full sweep and also record the "
    "lasso pick for offline accuracy comparison.",
)
parser.add_argument(
    "--k-neighborhood",
    type=int,
    default=2,
    help="(ring_xy/lasso) Half-width of the K neighborhood scanned "
    "classically around the LASSO ballpark K_hat.",
)
parser.add_argument(
    "--k-hedge",
    type=int,
    default=1,
    help="(ring_xy/lasso) Number of top classical K to actually run "
    "the fixed-K QAOA on (hedge against a wrong classical pick). "
    "Default lowered to 1 for the time-budgeted N=7..12 run (one QAOA "
    "solve per experiment instead of two, no truncation of any solve); "
    "raise back to 2 for unconstrained final runs.",
)
parser.add_argument(
    "--k-min",
    type=int,
    default=1,
    help="(ring_xy) Smallest cardinality K considered (default 1; "
    "K=1 is a single-asset portfolio. K=0 is excluded: the XY "
    "mixer conserves Hamming weight, so its subspace is the "
    "single frozen state |0...0> with nothing to optimize).",
)
parser.add_argument(
    "--exact-enum-max-n",
    type=int,
    default=12,
    help="(ring_xy) Max N (assets) for the post-objective "
    "full-subset enumeration in the exact selection "
    "reference; above this only the cheap energy-optimal "
    "reference is computed. Set 0 to disable enumeration.",
)
parser.add_argument(
    "--selection-weighting",
    choices=["budget", "unit"],
    default="budget",
    help="(ring_xy) How the selection HUBO weights a chosen asset (TASK-03). "
    "'budget' (default): y_i enters at the share count n_i=budget/price_i, so "
    "selection energy = cost(n.*y) and tracks post-allocation quality (K=1 "
    "inversion fixed at the source; K>=2 is a principled proxy). 'unit': the "
    "original price-blind y_i in {0,1} score (energy and post_objective are not "
    "co-monotone; kept for reproducibility of pre-fix results).",
)
parser.add_argument(
    "--experiments-json",
    default="experiments_data.json",
    help="Path to the experiments dataset JSON (default "
    "experiments_data.json; use a ring-XY-only dataset such as "
    "experiments_data_ring_xy.json for N>=7 problems).",
)
parser.add_argument(
    "--force",
    action="store_true",
    help="Re-run every experiment in the range and write a fresh "
    "timestamped results file even if an id already exists in an "
    "earlier results/ file for this method (bypasses the skip).",
)
args = parser.parse_args()

# Load experiments data
experiments = None
with open(args.experiments_json, "r") as f:
    experiments = list(json.load(f)["data"])

# Validate the requested experiment range (0-based, inclusive on both ends)
total_experiments = len(experiments)
if args.start < 0 or args.end < args.start or args.end >= total_experiments:
    print(
        f"Error: Invalid experiment range. Require 0 <= start <= end <= {total_experiments - 1}, "
        f"got start={args.start}, end={args.end}"
    )
    sys.exit(1)

#    The name of the SciPy optimizer to use. Must be one of:
#    'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B',
#    'TNC', 'COBYLA', 'COBYQA', 'SLSQP', 'trust-constr' (not reasonable for unconstrained optimization)),
#    'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact'
# Non-scipy optimizers:
#    'CMAES' (requires cma package to be installed)
# Order to try: 'COBYLA', 'SLSQP', 'Powell', 'CG', 'Nelder-Mead', 'L-BFGS-B'

classical_optimizer = "CMAES"
lambda_budget = 1.0

# Store all results in results/, with a per-run timestamped filename that embeds
# the experiment range, so re-running a range does not overwrite earlier runs.
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if args.method == "ring_xy":
    output_file = os.path.join(
        results_dir, f"ring_xy_exp_{timestamp}_{args.start}_{args.end}.json"
    )
    previous_prefix = "ring_xy_exp_"
else:
    output_file = os.path.join(
        results_dir,
        f"portfolio_optimization_exp_{classical_optimizer}_{str(lambda_budget)}_{timestamp}_{args.start}_{args.end}.json",
    )
    previous_prefix = f"portfolio_optimization_exp_{classical_optimizer}"

# Find earlier result files for this method (to skip already-processed
# experiments). Skipped entirely under --force, which re-runs the range.
previous_output_files = (
    []
    if args.force
    else [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.startswith(previous_prefix)
    ]
)

# Process the requested inclusive experiment range directly
start_idx = args.start
end_idx = args.end + 1  # exclusive upper bound for slicing

print(
    f"Processing experiments {args.start} to {args.end} (total {end_idx - start_idx})"
)

# Load existing results if file exists
all_existing_results = {}
for file in previous_output_files:
    if os.path.exists(file):
        with open(file, "r") as f:
            all_existing_results.update(json.load(f))

existing_results = {}
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        existing_results = json.load(f)

# Process only the experiments for this batch
for i, experiment in enumerate(experiments[start_idx:end_idx]):
    experiment_id = start_idx + i

    # Skip if already processed (unless --force re-runs everything)
    if not args.force and str(experiment_id) in all_existing_results:
        print(f"Skipping experiment {experiment_id} (already processed)")
        continue

    print(f"Processing experiment {experiment_id}")
    results_for_experiment = {}
    stocks = experiment["stocks"]
    start = experiment["start"]
    end = experiment["end"]
    risk_aversion = 0.1
    max_qubits = 15
    budget = experiment["budget"]
    print(f"Budget: {budget}")

    # Fetch with retry so transient Yahoo rate-limiting doesn't masquerade as a
    # delisting. Genuinely dead tickers (e.g. WBA) never recover and are dropped;
    # a throttled-but-healthy ticker (AAPL, V, MCD, ...) is retried, not dropped.
    fetched = fetch_close_prices(
        stocks, start=start, end=end, auto_adjust=False, progress=True
    )
    close = fetched.close
    dropped_tickers = fetched.dropped
    if dropped_tickers:
        print(
            f"Dropping unavailable tickers for experiment {experiment_id}: {dropped_tickers} "
            f"(delisted={fetched.delisted}, transient/unavailable={fetched.unavailable})"
        )
    if close.shape[1] < 2:
        print(
            f"Skipping experiment {experiment_id}: only {close.shape[1]} valid ticker(s) "
            f"after dropping {dropped_tickers}"
        )
        existing_results[str(experiment_id)] = {
            "error": "insufficient_valid_tickers",
            "dropped_tickers": dropped_tickers,
            "delisted": fetched.delisted,
            "unavailable": fetched.unavailable,
            "requested_stocks": [str(s) for s in stocks],
        }
        with open(output_file, "w") as f:
            json.dump(existing_results, f, indent=4)
        continue

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

    # Build the active solver and derive its qubit metadata. The classical
    # continuous baselines run through whichever solver is active; the
    # integer-HUBO exact baseline (below) is HOPO-only — its log-encoding needs
    # far more qubits than the ring-XY one-qubit-per-asset circuit, so
    # solve_exactly is infeasible for the larger ring-XY problems.
    if args.method == "hopo":
        portfolio_hubo = HigherOrderPortfolioQAOA(
            stocks=stocks,
            prices_now=prices_now,
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            budget=budget,
            max_qubits=max_qubits,
            coskewness_tensor=coskewness_tensor,
            cokurtosis_tensor=cokurtosis_tensor,
            log_encoding=True,
            risk_aversion=risk_aversion,
            strict_budget_constraint=False,
            lambda_budget=lambda_budget,
        )
        baseline_solver = portfolio_hubo
        assets_to_qubits = portfolio_hubo.get_assets_to_qubits()
        n_qubits = portfolio_hubo.get_n_qubits()
        n_layers = portfolio_hubo.get_layers()
    else:
        # Ring-XY: one qubit per asset, no log-encoding / budget-penalty qubits.
        ring_xy_solver = RingXYCardinalityQAOA(
            stocks=stocks,
            prices_now=prices_now,
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            budget=budget,
            coskewness_tensor=coskewness_tensor,
            cokurtosis_tensor=cokurtosis_tensor,
            risk_aversion=risk_aversion,
            selection_weighting=args.selection_weighting,
        )
        baseline_solver = ring_xy_solver
        assets_to_qubits = {str(s): [i] for i, s in enumerate(stocks)}
        n_qubits = ring_xy_solver.num_assets
        n_layers = min(8, n_qubits)  # matches build_cardinality_circuit default

    weights, allocation, value, left_overs = (
        baseline_solver.solve_with_continuous_variables()
    )

    continuous_variables_solution = {
        "weights": weights,
        "allocation": allocation,
        "value": value,
        "left_overs": left_overs,
    }

    continuous_variables_solution_unconstrained = None

    if coskewness_tensor is not None and cokurtosis_tensor is not None:
        weights, allocation, value, left_overs = (
            baseline_solver.solve_with_continuous_variables_unconstrained()
        )

        continuous_variables_solution_unconstrained = {
            "weights": weights,
            "allocation": allocation,
            "value": value,
            "left_overs": left_overs,
        }
    # Integer-HUBO exact baseline: HOPO only. The ring-XY method writes its own
    # selection_exact_solution instead; this integer log-encoding is exponential
    # for the larger ring-XY problems (a 7-asset problem needs 36+ qubits).
    exact_solution = None
    if args.method == "hopo":
        spectrum_is_partial = False
        try:
            (
                smallest_eigenvalues,
                smallest_bitstrings,
                first_excited_energy,
                optimized_portfolio,
                second_optimized_portfolio,
                eigenvalues,
                result1,
                result2,
            ) = portfolio_hubo.solve_exactly()

        except Exception as e:
            print(f"Error: {e}")
            print("Trying different classical eigenvalue solver")

            (
                smallest_eigenvalues,
                smallest_bitstrings,
                first_excited_energy,
                optimized_portfolio,
                second_optimized_portfolio,
                eigenvalues,
                result1,
                result2,
            ) = portfolio_hubo.solve_exactly_with_lobpcg()
            # lobpcg returns only its few smallest eigenvalues, not the full
            # spectrum, so max(spectrum) is not the true spectral maximum
            spectrum_is_partial = True

        exact_solution = {
            "smallest_eigenvalues": smallest_eigenvalues,
            "smallest_bitstrings": [
                "".join([str(i) for i in bits]) for bits in smallest_bitstrings
            ],
            "first_excited_energy": first_excited_energy,
            "optimized_portfolio": optimized_portfolio,
            "second_optimized_portfolio": second_optimized_portfolio,
            "spectrum": eigenvalues,
            "spectrum_is_partial": spectrum_is_partial,
            "result_with_budget": result1,
            "result_with_budget_excited": result2,
        }

        for key, value in exact_solution.items():
            if key != "spectrum":
                print(f"{key}: {value}")

    # Shared hyperparams (n_qubits / layers / assets_to_qubits set per method above)
    hyperparams = {
        "stocks": [str(s) for s in stocks],
        "start": start,
        "end": end,
        "risk_aversion": risk_aversion,
        "n_qubits": n_qubits,
        "budget": budget,
        "log_encoding": True,
        "layers": n_layers,
        "prices_now": {str(k): float(v) for k, v in prices_now.items()},
        "assets_to_qubits": {str(k): v for k, v in assets_to_qubits.items()},
        "optimizer": classical_optimizer,
        "lambda_budget": lambda_budget,
    }

    # Shared baselines written for either method
    results_for_experiment["hyperparams"] = hyperparams
    results_for_experiment["continuous_variables_solution"] = (
        continuous_variables_solution
    )
    results_for_experiment["continuous_variables_solution_unconstrained"] = (
        continuous_variables_solution_unconstrained
    )
    results_for_experiment["exact_solution"] = exact_solution

    if args.method == "hopo":
        # Original paper method: integer-HUBO + raw QAOA.
        if classical_optimizer == "CMAES":
            (
                two_most_probable_states,
                final_expectation_value,
                params,
                total_steps,
                states_probs,
                optimized_portfolios,
                training_history,
                objective_values,
                result1,
            ) = portfolio_hubo.solve_with_qaoa_cma_es()

        else:
            (
                two_most_probable_states,
                final_expectation_value,
                params,
                total_steps,
                states_probs,
                optimized_portfolios,
                training_history,
                objective_values,
                result1,
            ) = portfolio_hubo.solve_with_qaoa_scipy(optimizer=classical_optimizer)

        qaoa_solution = {
            "two_most_probable_states": two_most_probable_states,
            "final_expectation_value": float(final_expectation_value),
            "params": params.tolist(),
            "total_steps": total_steps,
            "states_probs": [float(v) for v in states_probs],
            "optimized_portfolios": optimized_portfolios,
            "training_history": training_history,
            "objective_values": objective_values,
            "result_with_budget": result1,
        }

        for key, value in qaoa_solution.items():
            if key != "training_history":
                print(f"{key}: {value}")

        results_for_experiment["qaoa_solution"] = qaoa_solution

    else:
        # Cardinality-selection QAOA: choose K (classical LASSO selector by
        # default; brute-force sweep available), then classical allocator on the
        # picked subset. ring_xy_solver was already built in the solver-build
        # block above (it also produced the shared continuous baseline).

        # Exact selection reference (energy-optimal per-K + post-objective ceiling).
        # Cheap: enumerates the 2^N selection-Hamiltonian diagonal (one qubit/asset).
        # Computed before the QAOA loop so a later QAOA failure still leaves the
        # reference, and so its cached diagonal primes the per-K ratios. Additive:
        # does NOT touch cardinality_qaoa_solution or its best_K rule.
        try:
            selection_exact = ring_xy_solver.solve_selection_exactly(
                k_min=args.k_min, enumeration_max_n=args.exact_enum_max_n
            )
        except Exception as e:
            print(f"Exact selection reference failed: {e}")
            selection_exact = {"error": str(e)}
        results_for_experiment["selection_exact_solution"] = selection_exact

        cardinality_qaoa_solution = {
            "per_K": [],
            "best_K": None,
            "k_selector": args.k_selector,
            "selection_weighting": args.selection_weighting,
        }
        N_assets = ring_xy_solver.num_assets

        # Build the per-K result entry by running the (unchanged) fixed-K QAOA.
        def run_qaoa_at(K):
            card_result = ring_xy_solver.solve_with_qaoa_cardinality(K)
            return {
                "K": card_result["K"],
                "layers": card_result["layers"],
                "final_expectation_value": card_result["final_expectation_value"],
                "approximation_ratio_subspace": card_result[
                    "approximation_ratio_subspace"
                ],
                "approximation_ratio_global": card_result["approximation_ratio_global"],
                "selection_energy_min_subspace": card_result[
                    "selection_energy_min_subspace"
                ],
                "selection_energy_max_subspace": card_result[
                    "selection_energy_max_subspace"
                ],
                "selection_energy_dicke_mean": card_result[
                    "selection_energy_dicke_mean"
                ],
                "selection_energy_global_min": card_result[
                    "selection_energy_global_min"
                ],
                "selection_energy_global_max": card_result[
                    "selection_energy_global_max"
                ],
                "selected_indices": list(card_result["selected_indices"]),
                "selected_stocks": [str(s) for s in card_result["selected_stocks"]],
                "selection_bitstring": card_result["selection_bitstring"],
                "raw_top_in_subspace": card_result["raw_top_in_subspace"],
                "subspace_leakage": card_result["subspace_leakage"],
                "top_in_subspace_prob": card_result["top_in_subspace_prob"],
                "iterations": card_result["iterations"],
                "allocation": {str(k): v for k, v in card_result["allocation"].items()},
                "realized_budget": card_result["realized_budget"],
                "leftover_budget": card_result["leftover_budget"],
                "post_objective": card_result["post_objective"],
                "infeasible": card_result["infeasible"],
                "infeasible_reason": card_result["infeasible_reason"],
                "effective_K": card_result["effective_K"],
                "training_history": card_result["training_history"],
            }

        # Decide which K to hand to the expensive QAOA.
        if args.k_selector in ("lasso", "both"):
            sel = ring_xy_solver.select_K_classically(
                neighborhood=args.k_neighborhood, k_min=args.k_min
            )
            cardinality_qaoa_solution["k_hat"] = sel["k_hat"]
            cardinality_qaoa_solution["classical_K_scan"] = sel["scan"]
            cardinality_qaoa_solution["lasso_support_path"] = sel["support_path"]
            print(
                f"--- LASSO localizer: K_hat={sel['k_hat']}, "
                f"classical ranking={sel['ranked_K']} ---"
            )

        if args.k_selector == "sweep":
            K_list = list(range(args.k_min, N_assets + 1))
        elif args.k_selector == "lasso":
            K_list = sel["ranked_K"][: args.k_hedge]
            if not K_list:  # nothing classically feasible -> fall back to k_min
                K_list = [min(max(args.k_min, 1), N_assets)]
        else:  # "both": full sweep is the ground truth; record what lasso would pick
            K_list = list(range(args.k_min, N_assets + 1))
            cardinality_qaoa_solution["lasso_picked_K"] = sel["ranked_K"][
                : args.k_hedge
            ]

        cardinality_qaoa_solution["qaoa_K_run"] = list(K_list)

        # best_K is chosen by post-allocation `post_objective`, NOT by selection
        # energy / approximation ratio, and this is deliberate (TASK-03). The
        # selection HUBO scores subsets at unit weight (binary y_i), but
        # portfolio quality is judged only after the continuous allocator runs at
        # budget-scaled share counts. The two are not co-monotone, so the
        # energy-minimizing selection at a given K can be the *worst* portfolio
        # after allocation (id 1: the K=1 QAOA finds NKE perfectly,
        # approximation_ratio_subspace ~ 0.9999, yet post_objective ~ -136,281,
        # the worst single asset; the best single-asset portfolio is TRV at
        # ~ -2,106, structurally unreachable because selection energy ranks NKE
        # above TRV). The per-K `approximation_ratio_*` fields therefore measure
        # QAOA *energy convergence*, not portfolio quality. See
        # RingXYCardinalityQAOA._approximation_ratios / solve_selection_exactly.
        best_post_obj = None
        for K in K_list:
            print(f"--- Cardinality QAOA: K={K} of N={N_assets} ---")
            try:
                per_K_entry = run_qaoa_at(K)
            except Exception as e:
                print(f"Cardinality QAOA failed at K={K}: {e}")
                cardinality_qaoa_solution["per_K"].append({"K": K, "error": str(e)})
                continue
            cardinality_qaoa_solution["per_K"].append(per_K_entry)
            if (
                not per_K_entry["infeasible"]
                and per_K_entry["post_objective"] is not None
            ):
                if (
                    best_post_obj is None
                    or per_K_entry["post_objective"] > best_post_obj
                ):
                    best_post_obj = per_K_entry["post_objective"]
                    cardinality_qaoa_solution["best_K"] = per_K_entry["K"]

        results_for_experiment["cardinality_qaoa_solution"] = cardinality_qaoa_solution

    # Add to existing_results
    existing_results[str(experiment_id)] = results_for_experiment

    # Write updated results to file after each experiment (to avoid losing progress)
    with open(output_file, "w") as f:
        json.dump(existing_results, f, indent=4)

print(
    f"Experiments {args.start} to {args.end} completed. Results saved to {output_file}"
)
