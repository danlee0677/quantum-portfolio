"""Cardinality-selection hybrid QAOA (the ring XY-mixer contribution).

Stand-alone from the paper's integer-HUBO method in
``portfolio_hubo_qaoa_light.py``. This class solves a y_i in {0,1} HUBO that
picks K of N stocks (one qubit per asset, no budget penalty), prepares a Dicke
initial state, mixes with a ring XY-mixer (``qml.qaoa.xy_mixer`` on a cycle
graph), and then runs a classical integer-program allocator on the chosen
K-subset.

It shares no algorithm code with ``HigherOrderPortfolioQAOA``: the two small
helpers it needs (``get_objective_value`` for objective evaluation and
``cma_result_to_dict`` for result formatting) are kept here as their own copies.
Shared utilities live in ``utils.py`` and the classical baselines
(``HigherMomentPortfolioOptimizer``, ``EfficientFrontier``,
``DiscreteAllocation``) are imported, not reimplemented.
"""

import cma
import networkx as nx
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
from pypfopt import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation

from portfolio_higher_moments_classical import HigherMomentPortfolioOptimizer
from utils import dicke_state_vector, hamming_weight_indices, normalize_linear_combination


class RingXYCardinalityQAOA:

    def __init__(self,
                 stocks,
                 prices_now,
                 expected_returns,
                 covariance_matrix,
                 budget,
                 coskewness_tensor=None,
                 cokurtosis_tensor=None,
                 risk_aversion=3):
        # Implementation assumes that stocks and other data are ordered to match.
        # The cardinality method needs only these inputs: no log-encoding qubit
        # map, no budget-penalty construction, no integer HUBO Hamiltonian.
        self.stocks = stocks
        self.prices_now = prices_now
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.budget = budget
        self.coskewness_tensor = coskewness_tensor
        self.cokurtosis_tensor = cokurtosis_tensor
        self.risk_aversion = risk_aversion
        self.num_assets = len(expected_returns)

        assert len(stocks) == len(expected_returns)
        assert len(expected_returns) == len(covariance_matrix)
        assert len(expected_returns) == len(covariance_matrix[0])
        assert risk_aversion > 0

    def cma_result_to_dict(self, result):
        """ Converts CMAEvolutionStrategyResult to a pure Python dictionary. """
        return {
            "xbest": result.xbest.tolist(),
            "fbest": result.fbest,
            "evals_best": int(result.evals_best),  # Convert np.int64 to int
            "evaluations": result.evaluations,
            "iterations": result.iterations,
            "xfavorite": result.xfavorite.tolist(),
            "stds": result.stds.tolist(),
            "stop": result.stop  # Already a dictionary
        }

    def get_objective_value(self, optimized_portfolio):
        for stock in self.stocks:
            if stock not in optimized_portfolio:
                optimized_portfolio[stock] = 0

        objective_value = 0
        for i in range(len(self.stocks)):
            objective_value -= optimized_portfolio[self.stocks[i]]*self.expected_returns[i]

        for i in range(len(self.stocks)):
            for j in range(len(self.stocks)):
                objective_value += (self.risk_aversion/2)*optimized_portfolio[self.stocks[i]]*optimized_portfolio[self.stocks[j]]*self.covariance_matrix[i][j]

        if self.coskewness_tensor is not None:
            for i in range(len(self.stocks)):
                for j in range(len(self.stocks)):
                    for k in range(len(self.stocks)):
                        objective_value -= (self.risk_aversion/6)*optimized_portfolio[self.stocks[i]]*optimized_portfolio[self.stocks[j]]*optimized_portfolio[self.stocks[k]]*self.coskewness_tensor[i][j][k]

        if self.cokurtosis_tensor is not None:
            for i in range(len(self.stocks)):
                for j in range(len(self.stocks)):
                    for k in range(len(self.stocks)):
                        for l in range(len(self.stocks)):
                            objective_value += (self.risk_aversion/24)*optimized_portfolio[self.stocks[i]]*optimized_portfolio[self.stocks[j]]*optimized_portfolio[self.stocks[k]]*optimized_portfolio[self.stocks[l]]*self.cokurtosis_tensor[i][j][k][l]

        return -objective_value

    # ------------------------------------------------------------------
    # Cardinality-selection hybrid: y_i ∈ {0,1} HUBO + classical allocator
    # ------------------------------------------------------------------

    def construct_selection_hubo_bin(self, K):
        # Equal-weight selection score over y_i ∈ {0,1}, one qubit per asset.
        # Kurtosis coefficient is risk_aversion/24 to match get_objective_value
        # and the classical baseline (not the risk_aversion**2/24 used in
        # construct_cost_hubo_int line 147, which disagrees with both).
        self.selection_K = K
        self.selection_n_qubits = self.num_assets

        hubo_bin = {}
        for i in range(self.num_assets):
            hubo_bin[((self.stocks[i], i),)] = -self.expected_returns[i]

        for i in range(self.num_assets):
            for j in range(self.num_assets):
                cov = (self.risk_aversion/2)*self.covariance_matrix[i][j]
                hubo_bin[((self.stocks[i], i), (self.stocks[j], j))] = cov

        if self.coskewness_tensor is not None:
            for i in range(self.num_assets):
                for j in range(self.num_assets):
                    for k in range(self.num_assets):
                        skew = (self.risk_aversion/6)*self.coskewness_tensor[i][j][k]
                        hubo_bin[((self.stocks[i], i), (self.stocks[j], j), (self.stocks[k], k))] = -skew

        if self.cokurtosis_tensor is not None:
            for i in range(self.num_assets):
                for j in range(self.num_assets):
                    for k in range(self.num_assets):
                        for l in range(self.num_assets):
                            kurt = (self.risk_aversion/24)*self.cokurtosis_tensor[i][j][k][l]
                            hubo_bin[((self.stocks[i], i), (self.stocks[j], j), (self.stocks[k], k), (self.stocks[l], l))] = kurt

        simplified = {}
        for bin_var, coeff in hubo_bin.items():
            key = frozenset(bin_var)
            simplified[key] = simplified.get(key, 0) + coeff
        self.selection_hubo_bin_simplified = simplified

        ham = 0
        for bin_var_set, coeff in simplified.items():
            bin_var = list(bin_var_set)
            if len(bin_var) == 1:
                q0 = bin_var[0][1]
                ham += (coeff/2)*(qml.Identity(q0) - qml.PauliZ(q0))
            elif len(bin_var) == 2:
                q0, q1 = bin_var[0][1], bin_var[1][1]
                ham += (coeff/4)*(qml.Identity(q0) - qml.PauliZ(q0)) @ (qml.Identity(q1) - qml.PauliZ(q1))
            elif len(bin_var) == 3:
                q0, q1, q2 = bin_var[0][1], bin_var[1][1], bin_var[2][1]
                ham += (coeff/8)*(qml.Identity(q0) - qml.PauliZ(q0)) @ (qml.Identity(q1) - qml.PauliZ(q1)) @ (qml.Identity(q2) - qml.PauliZ(q2))
            elif len(bin_var) == 4:
                q0, q1, q2, q3 = bin_var[0][1], bin_var[1][1], bin_var[2][1], bin_var[3][1]
                ham += (coeff/16)*(qml.Identity(q0) - qml.PauliZ(q0)) @ (qml.Identity(q1) - qml.PauliZ(q1)) @ (qml.Identity(q2) - qml.PauliZ(q2)) @ (qml.Identity(q3) - qml.PauliZ(q3))
        self.selection_hamiltonian = normalize_linear_combination(ham)

    def _slice_problem(self, selected_indices):
        sel_stocks = [self.stocks[i] for i in selected_indices]
        sel_prices = pd.Series({self.stocks[i]: float(self.prices_now[self.stocks[i]]) for i in selected_indices})
        sel_mu = np.array([float(self.expected_returns[i]) for i in selected_indices])
        sel_sigma = np.array([[float(self.covariance_matrix[i][j]) for j in selected_indices] for i in selected_indices])
        sel_S = None
        sel_K = None
        if self.coskewness_tensor is not None:
            sel_S = np.array([[[float(self.coskewness_tensor[i][j][k]) for k in selected_indices] for j in selected_indices] for i in selected_indices])
        if self.cokurtosis_tensor is not None:
            sel_K = np.array([[[[float(self.cokurtosis_tensor[i][j][k][l]) for l in selected_indices] for k in selected_indices] for j in selected_indices] for i in selected_indices])
        return sel_stocks, sel_prices, sel_mu, sel_sigma, sel_S, sel_K

    def _allocate_on_subset(self, selected_indices):
        sel_stocks, sel_prices, sel_mu, sel_sigma, sel_S, sel_K_tensor = self._slice_problem(selected_indices)

        min_unit_cost = float(sel_prices.sum())
        if min_unit_cost > self.budget:
            return {
                "allocation": {self.stocks[i]: 0 for i in range(self.num_assets)},
                "realized_budget": 0.0,
                "leftover_budget": float(self.budget),
                "post_objective": None,
                "infeasible": True,
                "infeasible_reason": "min_unit_cost_exceeds_budget",
            }

        try:
            if sel_S is not None and sel_K_tensor is not None:
                hef = HigherMomentPortfolioOptimizer(
                    sel_stocks, sel_mu, sel_sigma, sel_S, sel_K_tensor,
                    risk_aversion=self.risk_aversion)
                weights = hef.optimize_portfolio_with_higher_moments_unconstrained()
            else:
                ef = EfficientFrontier(sel_mu, sel_sigma)
                w_idx = ef.max_quadratic_utility(risk_aversion=self.risk_aversion)
                weights = {sel_stocks[k]: float(v) for k, v in w_idx.items()}

            allocator = DiscreteAllocation(weights, sel_prices, self.budget)
            allocation, leftover = allocator.lp_portfolio()
        except Exception as e:
            print(f"DiscreteAllocation/optimizer failed on subset: {e}")
            return {
                "allocation": {self.stocks[i]: 0 for i in range(self.num_assets)},
                "realized_budget": 0.0,
                "leftover_budget": float(self.budget),
                "post_objective": None,
                "infeasible": True,
                "infeasible_reason": str(e),
            }

        full_alloc = {self.stocks[i]: 0 for i in range(self.num_assets)}
        realized = 0.0
        for s, amount in allocation.items():
            full_alloc[s] = int(amount)
            realized += float(self.prices_now[s]) * int(amount)

        post_obj = float(self.get_objective_value(dict(full_alloc)))

        return {
            "allocation": full_alloc,
            "realized_budget": float(realized),
            "leftover_budget": float(leftover),
            "post_objective": post_obj,
            "infeasible": False,
        }

    def build_cardinality_circuit(self, K, device_name="lightning.qubit", layers=None):
        """
        Construct the cardinality-selection QAOA circuit (Dicke + ring XY)
        without running optimization. Returns
        (qaoa_circuit, qaoa_probs_circuit, layers_used, n_qubits).
        """
        N = self.num_assets
        assert 1 <= K <= N, f"K={K} must be in [1, N={N}]"

        self.construct_selection_hubo_bin(K)
        cost_hamiltonian = self.selection_hamiltonian
        mixer_hamiltonian = qml.qaoa.xy_mixer(nx.cycle_graph(N))

        layers_used = layers if layers is not None else min(8, N)
        dicke_vec = dicke_state_vector(N, K)
        dev = qml.device(device_name, wires=N)

        def qaoa_layer(gamma, beta):
            qml.qaoa.cost_layer(gamma, cost_hamiltonian)
            qml.qaoa.mixer_layer(beta, mixer_hamiltonian)

        @qml.qnode(dev)
        def qaoa_circuit(params):
            gammas = params[:layers_used]
            betas = params[layers_used:]
            qml.StatePrep(dicke_vec, wires=range(N))
            qml.layer(qaoa_layer, layers_used, gammas, betas)
            return qml.expval(cost_hamiltonian)

        @qml.qnode(dev)
        def qaoa_probs_circuit(params):
            gammas = params[:layers_used]
            betas = params[layers_used:]
            qml.StatePrep(dicke_vec, wires=range(N))
            qml.layer(qaoa_layer, layers_used, gammas, betas)
            return qml.probs()

        return qaoa_circuit, qaoa_probs_circuit, layers_used, N

    def solve_with_qaoa_cardinality(self, K, layers=None, maxiter=None):
        N = self.num_assets
        assert 1 <= K <= N, f"K={K} must be in [1, N={N}]"

        qaoa_circuit, qaoa_probs_circuit, sel_layers, _ = self.build_cardinality_circuit(K, layers=layers)
        if maxiter is None:
            maxiter = 800 if N <= 13 else 300

        def objective_function(params):
            return float(qaoa_circuit(params))

        initial_params = np.pi*np.random.rand(2*sel_layers)
        es = cma.CMAEvolutionStrategy(initial_params, sigma0=0.1, options={"maxiter": maxiter})
        result = es.optimize(objective_function)
        optimized_params = result.result.xbest
        final_expectation_value = float(qaoa_circuit(optimized_params))
        probs = qaoa_probs_circuit(optimized_params)

        valid_indices = hamming_weight_indices(N, K)
        valid_mass = float(sum(probs[i] for i in valid_indices))
        leakage = max(0.0, 1.0 - valid_mass)
        raw_top = int(np.argmax(probs))
        raw_in_subspace = bin(raw_top).count("1") == K
        top_in_subspace = max(valid_indices, key=lambda i: float(probs[i]))

        bitstring = format(top_in_subspace, f"0{N}b")
        selected_indices = [q for q in range(N) if bitstring[q] == "1"]
        selected_stocks = [self.stocks[q] for q in selected_indices]

        allocation_info = self._allocate_on_subset(selected_indices)

        return {
            "K": K,
            "layers": sel_layers,
            "final_expectation_value": final_expectation_value,
            "selected_indices": selected_indices,
            "selected_stocks": selected_stocks,
            "selection_bitstring": bitstring,
            "raw_top_in_subspace": bool(raw_in_subspace),
            "subspace_leakage": leakage,
            "top_in_subspace_prob": float(probs[top_in_subspace]),
            "optimized_params": optimized_params.tolist(),
            "iterations": int(result.result.iterations),
            "training_history": self.cma_result_to_dict(result.result),
            "allocation": allocation_info["allocation"],
            "realized_budget": allocation_info["realized_budget"],
            "leftover_budget": allocation_info["leftover_budget"],
            "post_objective": allocation_info["post_objective"],
            "infeasible": allocation_info["infeasible"],
            "infeasible_reason": allocation_info.get("infeasible_reason"),
        }
