import json
import numpy as np
import yfinance as yf
from coskweness_cokurtosis import coskewness, cokurtosis
from old.portfolio_hubo_qaoa import HigherOrderPortfolioQAOA
import os
from concurrent.futures import ProcessPoolExecutor
import traceback

experiments = None
with open("experiments_data.json", "r") as f:
    experiments = list(json.load(f)["data"])

output_file = "portfolio_optimization_results.json"

init_experiment = 0
# Load existing data if file exists
if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        existing_results = json.load(f)
    init_experiment = max([int(k) for k in existing_results.keys()]) + 1

def process_experiment(experiment_data):
    try:
        experiment_index, experiment = experiment_data
        results_for_experiment = {}
        stocks = experiment["stocks"]
        start = experiment["start"]
        end = experiment["end"]
        risk_aversion = 0.1
        max_qubits = 15
        budget = experiment["budget"]
        print(f"Processing experiment {experiment_index} with budget {budget}")

        data = yf.download(stocks, start=start, end=end)
        prices_now = data["Close"].iloc[-1]
            
        returns = data["Close"].pct_change().dropna()
        stocks = returns.columns

        numpy_returns = returns.to_numpy()
        expected_returns = returns.add(1).prod() ** (252 / len(returns))
        expected_returns = expected_returns.to_numpy()
        covariance_matrix = np.cov(numpy_returns, rowvar=False)*(252/len(returns))
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
                                                risk_aversion=risk_aversion)
        
        assets_to_qubits = portfolio_hubo.get_assets_to_qubits()
        
        weights, allocation, value, left_overs = portfolio_hubo.solve_with_continuous_variables()

        continuous_variables_solution = {
            "weights": weights,
            "allocation": allocation,
            "value": value,
            "left_overs": left_overs
        }

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

        print(f"Experiment {experiment_index} - Exact solution found")
        
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

        print(f"Experiment {experiment_index} - QAOA solution found")
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
            "assets_to_qubits": {str(k): v for k, v in assets_to_qubits.items()}
        }

        results_for_experiment["hyperparams"] = hyperparams
        results_for_experiment["continuous_variables_solution"] = continuous_variables_solution
        results_for_experiment["exact_solution"] = exact_solution
        results_for_experiment["qaoa_solution"] = qaoa_solution

        return experiment_index, results_for_experiment
    except Exception as e:
        print(f"Error in experiment {experiment_index}: {str(e)}")
        traceback.print_exc()
        return experiment_index, {"error": str(e)}

if __name__ == "__main__":
    # Determine the maximum number of workers based on CPU cores
    max_workers = min(os.cpu_count() - 1, len(experiments[init_experiment:]))

    # Create experiment indices
    experiment_data = [(i + init_experiment, exp) for i, exp in enumerate(experiments[init_experiment:])]

    # Load existing results once at the beginning
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_results = json.load(f)
    else:
        existing_results = {}

    def save_result(future):
        experiment_index, result = future.result()
        # Update results and save immediately
        existing_results[str(experiment_index)] = result
        with open(output_file, 'w') as f:
            json.dump(existing_results, f, indent=4)
        print(f"Experiment {experiment_index} results saved to {output_file}")

    # Process experiments in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for exp_data in experiment_data:
            future = executor.submit(process_experiment, exp_data)
            future.add_done_callback(save_result)
            futures = [future for future in futures if not future.done()]

    print(f"All experiments completed and saved to {output_file}")