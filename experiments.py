import json
import yfinance as yf
from coskweness_cokurtosis import coskewness, cokurtosis
from portfolio_hubo_qaoa_light import HigherOrderPortfolioQAOA
import os
import sys
import argparse
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run portfolio optimization experiments in batches')
parser.add_argument('batch_num', type=int, help='Batch number to process (0-indexed)')
parser.add_argument('total_batches', type=int, help='Total number of batches')
args = parser.parse_args()

# Validate arguments
if args.batch_num < 0 or args.total_batches <= 0 or args.batch_num >= args.total_batches:
    print(f"Error: Invalid batch parameters. batch_num must be between 0 and {args.total_batches-1}")
    sys.exit(1)

# Load experiments data
experiments = None
with open("experiments_data.json", "r") as f:
    experiments = list(json.load(f)["data"])

#    The name of the SciPy optimizer to use. Must be one of:
#    'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B',
#    'TNC', 'COBYLA', 'COBYQA', 'SLSQP', 'trust-constr' (not reasonable for unconstrained optimization)), 
#    'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact'
# Non-scipy optimizers:
#    'CMAES' (requires cma package to be installed)
# Order to try: 'COBYLA', 'SLSQP', 'Powell', 'CG', 'Nelder-Mead', 'L-BFGS-B'

classical_optimizer = "CMAES"
lambda_budget = 0.001

output_file = f"portfolio_optimization_batch_{classical_optimizer}_{str(lambda_budget)}_{args.batch_num}.json"

# Find files with portfolio_optimization_results_batch_ in the name
previous_output_files = [f for f in os.listdir() if f"portfolio_optimization_results_batch_{classical_optimizer}" in f]

# Calculate which experiments to process in this batch
total_experiments = len(experiments)
batch_size = total_experiments // args.total_batches
remainder = total_experiments % args.total_batches

# Distribute remainder across batches
start_idx = args.batch_num * batch_size + min(args.batch_num, remainder)
end_idx = start_idx + batch_size + (1 if args.batch_num < remainder else 0)

print(f"Processing batch {args.batch_num+1}/{args.total_batches}: experiments {start_idx} to {end_idx-1} (total {end_idx-start_idx})")

# Load existing results if file exists
all_existing_results = {}
for file in previous_output_files:
    if os.path.exists(file):
        with open(file, 'r') as f:
            all_existing_results.update(json.load(f))

existing_results = {}
if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        existing_results = json.load(f)

# Process only the experiments for this batch
for i, experiment in enumerate(experiments[start_idx:end_idx]):
    experiment_id = start_idx + i
    
    # Skip if already processed
    if str(experiment_id) in all_existing_results:
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

    data = yf.download(stocks, start=start, end=end)
    prices_now = data["Close"].iloc[-1]
    returns = data["Close"].pct_change(fill_method=None).dropna(how="any")
    stocks = returns.columns
    numpy_returns = returns.to_numpy()

    expected_returns = mean_historical_return(returns, returns_data=True, compounding=False).to_numpy()
    covariance_matrix = sample_cov(returns, returns_data=True).to_numpy()
    coskewness_tensor = coskewness(numpy_returns)
    cokurtosis_tensor = cokurtosis(numpy_returns)

    portfolio_hubo = HigherOrderPortfolioQAOA(stocks=stocks,
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
                                            lambda_budget=lambda_budget)
    
    assets_to_qubits = portfolio_hubo.get_assets_to_qubits()
    
    weights, allocation, value, left_overs = portfolio_hubo.solve_with_continuous_variables()

    continuous_variables_solution = {
        "weights": weights,
        "allocation": allocation,
        "value": value,
        "left_overs": left_overs
    }
    
    continuous_variables_solution_unconstrained = None

    if coskewness_tensor is not None and cokurtosis_tensor is not None:
        
        weights, allocation, value, left_overs = portfolio_hubo.solve_with_continuous_variables_unconstrained()

        continuous_variables_solution_unconstrained = {
            "weights": weights,
            "allocation": allocation,
            "value": value,
            "left_overs": left_overs
        }
    if True:
        try:
            (
                smallest_eigenvalues, 
                smallest_bitstrings, 
                first_excited_energy, 
                optimized_portfolio, 
                second_optimized_portfolio,
                eigenvalues,
                result1, 
                result2
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
                result2
            ) = portfolio_hubo.solve_exactly_with_lobpcg()

        exact_solution = {
            "smallest_eigenvalues": smallest_eigenvalues,
            "smallest_bitstrings": ["".join([str(i) for i in bits]) for bits in smallest_bitstrings],
            "first_excited_energy": first_excited_energy,
            "optimized_portfolio": optimized_portfolio,
            "second_optimized_portfolio": second_optimized_portfolio,
            "spectrum": eigenvalues,
            "result_with_budget": result1,
            "result_with_budget_excited": result2
        }

        for key, value in exact_solution.items():
            if key != "spectrum":
                print(f"{key}: {value}")
    
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
            result1
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
            result1
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
        "result_with_budget": result1
    }

    for key, value in qaoa_solution.items():
        if key != "training_history":
            print(f"{key}: {value}")
 
    n_qubits = portfolio_hubo.get_n_qubits()
    n_layers = portfolio_hubo.get_layers()
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
        "lambda_budget": lambda_budget
    }

    results_for_experiment["hyperparams"] = hyperparams
    results_for_experiment["continuous_variables_solution"] = continuous_variables_solution
    results_for_experiment["continuous_variables_solution_unconstrained"] = continuous_variables_solution_unconstrained
    results_for_experiment["exact_solution"] = exact_solution
    results_for_experiment["qaoa_solution"] = qaoa_solution

    # Add to existing_results
    existing_results[str(experiment_id)] = results_for_experiment

    # Write updated results to file after each experiment (to avoid losing progress)
    with open(output_file, 'w') as f:
        json.dump(existing_results, f, indent=4)

print(f"Batch {args.batch_num+1}/{args.total_batches} completed. Results saved to {output_file}")
