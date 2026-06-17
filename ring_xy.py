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
from pypfopt import EfficientFrontier, objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation

from portfolio_higher_moments_classical import HigherMomentPortfolioOptimizer
from utils import dicke_state_vector, hamming_weight_indices, normalize_linear_combination

# Total-weight floor below which a subset's continuous answer is "hold nothing".
# In that regime DiscreteAllocation.lp_portfolio is exactly tie-degenerate (each
# dollar spent raises the deviation term as much as it lowers the leftover term),
# so its output is an arbitrary LP vertex and must not be trusted. Absolute, not
# relative to max(weight): max ~ 0 IS the trigger case. Safe bounds: well above
# scipy's bound-clipping noise (~1e-9) and well below the smallest weight the
# discrete problem can even represent (min(price)/budget, ~3e-3 here).
ZERO_WEIGHT_EPS = 1e-6


class RingXYCardinalityQAOA:

    def __init__(self,
                 stocks,
                 prices_now,
                 expected_returns,
                 covariance_matrix,
                 budget,
                 coskewness_tensor=None,
                 cokurtosis_tensor=None,
                 risk_aversion=3,
                 selection_weighting="budget"):
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
        # How the selection HUBO weights a chosen asset (TASK-03):
        #   "budget" (default): y_i enters at the share count n_i = budget/price_i,
        #     so selection energy = cost(n .* y) and tracks post-allocation quality
        #     (K=1 inversion fixed at the source; K>=2 is a principled proxy).
        #   "unit": the original price-blind y_i in {0,1} score (energy and
        #     post_objective are not co-monotone; kept for reproducibility).
        # K-independent on purpose (see _selection_energy_diagonal / TASK-03 note).
        self.selection_weighting = selection_weighting

        assert len(stocks) == len(expected_returns)
        assert len(expected_returns) == len(covariance_matrix)
        assert len(expected_returns) == len(covariance_matrix[0])
        assert risk_aversion > 0
        assert selection_weighting in ("budget", "unit")

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
    # Classical continuous baselines (shared comparison point for both
    # methods). Transplanted verbatim from HigherOrderPortfolioQAOA so the
    # ring-XY runner can write the same baseline WITHOUT building the integer
    # log-encoded HUBO object — these go straight through
    # HigherMomentPortfolioOptimizer + DiscreteAllocation, which scale with the
    # asset count, not the budget-sized qubit encoding.
    # ------------------------------------------------------------------

    def solve_with_continuous_variables(self):
        if self.coskewness_tensor is None and self.cokurtosis_tensor is None:
            ef = EfficientFrontier(self.expected_returns, self.covariance_matrix)
            weights = ef.max_quadratic_utility(risk_aversion=self.risk_aversion)

            for asset, weight in weights.items():
                print(f"{self.stocks[asset]}: {weight:.2%}")

            allocator = DiscreteAllocation(weights, self.prices_now, self.budget)
            allocation, left_overs = allocator.lp_portfolio()
            print("Left over budget: ", left_overs)

            print("Optimized discrete allocation for mean and variance:")
            final_allocation = {}
            for asset, amount in allocation.items():
                final_allocation[self.stocks[asset]] = amount
                print(f"{self.stocks[asset]}: {amount}")

            value = self.get_objective_value(final_allocation)
            print("Maximized utility from continuous mean variance: ", value)

            weights = {self.stocks[asset]: weight for asset, weight in weights.items()}

            return weights, final_allocation, value, left_overs

        else:
            hef = HigherMomentPortfolioOptimizer(self.stocks,
                                                 self.expected_returns,
                                                 self.covariance_matrix,
                                                 self.coskewness_tensor,
                                                 self.cokurtosis_tensor,
                                                 risk_aversion=self.risk_aversion)
            weights = hef.optimize_portfolio_with_higher_moments()

            print("Optimized Weights (considering variance, skewness and kurtosis):")
            for asset, weight in weights.items():
                print(f"{asset}: {weight:.2%}")

            allocation, left_overs = hef.get_discrete_allocation(weights, self.prices_now, self.budget)
            print("Left over budget: ", left_overs)

            print("Optimized Discrete Allocation:")
            for asset, amount in allocation.items():
                print(f"{asset}: {amount}")

            for stock in self.stocks:
                if stock not in allocation:
                    allocation[stock] = 0

            value = self.get_objective_value(allocation)
            print("Maximized utility from continuous higher moments: ", value)

            return weights, allocation, value, left_overs

    def solve_with_continuous_variables_unconstrained(self):
        if self.coskewness_tensor is None and self.cokurtosis_tensor is None:
            raise ValueError("Unconstrained optimization is only possible with higher moments")
        else:
            hef = HigherMomentPortfolioOptimizer(self.stocks,
                                                 self.expected_returns,
                                                 self.covariance_matrix,
                                                 self.coskewness_tensor,
                                                 self.cokurtosis_tensor,
                                                 risk_aversion=self.risk_aversion)

            weights = hef.optimize_portfolio_with_higher_moments_unconstrained()

            print("Optimized Weights with unconstrained classical continuous variable (considering variance, skewness and kurtosis):")
            for asset, weight in weights.items():
                print(f"{asset}: {weight:.2%}")

            allocation, left_overs = hef.get_discrete_allocation(weights, self.prices_now, self.budget)
            print("Left over budget for unconstrained: ", left_overs)

            print("Optimized Discrete Allocation for unconstrained:")
            for asset, amount in allocation.items():
                print(f"{asset}: {amount}")

            for stock in self.stocks:
                if stock not in allocation:
                    allocation[stock] = 0

            value = self.get_objective_value(allocation)
            print("Maximized utility from continuous higher moments for unconstrained: ", value)

            return weights, allocation, value, left_overs

    # ------------------------------------------------------------------
    # Cardinality-selection hybrid: y_i ∈ {0,1} HUBO + classical allocator
    # ------------------------------------------------------------------

    def construct_selection_hubo_bin(self, K):
        # Selection score over y_i ∈ {0,1}, one qubit per asset.
        # NOTE (TASK-03): two weightings, chosen by self.selection_weighting:
        #   "budget" (default): each chosen asset enters at the share count
        #     n_i = budget/price_i, so this energy equals cost(n .* y) -- the
        #     portfolio cost at budget-scaled shares. Price-aware and degree-4
        #     correct, so the energy-minimizing subset tracks post-allocation
        #     quality: exact at K=1 (single-asset shares are exactly n_i), a
        #     principled proxy at K>=2 (the allocator distributes non-equally, so
        #     no fixed-weight HUBO is co-monotone with post_objective there).
        #   "unit": the original price-blind y_i score, where selection energy and
        #     post_objective are NOT co-monotone and the energy-min subset can be
        #     the worst portfolio after allocation.
        # Either way the pipeline still ranks K / best_K by post_objective (the
        # K>=2 safety net); see `_approximation_ratios` and `select_K_classically`.
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

        # Budget-weighting (TASK-03): replace y_i -> n_i*y_i with the share count
        # n_i = budget/price_i, turning this unit-weight score into cost(n .* y).
        # CRITICAL: scale the *raw* monomial keys here, which carry the term's
        # degree as tuple multiplicity, *before* the frozenset collapse below. The
        # i==j covariance key ((s_i,i),(s_i,i)) must pick up n_i**2; the frozenset
        # then dedupes it to {(s_i,i)} and merges it with the linear term
        # -mu_i*n_i, yielding exactly the linear+diagonal of cost(n .* y). Scaling
        # the simplified frozenset instead would apply n_i only once and silently
        # break every diagonal / repeated-index term.
        if self.selection_weighting == "budget":
            n = []
            for i in range(self.num_assets):
                p = float(self.prices_now[self.stocks[i]])
                assert np.isfinite(p) and p > 0, (
                    f"budget-weighting needs a positive finite price for "
                    f"{self.stocks[i]}, got {p}"
                )
                n.append(self.budget / p)
            for key in list(hubo_bin):
                factor = 1.0
                for (_stock, idx) in key:   # repetition => correct n_i**degree
                    factor *= n[idx]
                hubo_bin[key] *= factor

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

    def _selection_energy_diagonal(self):
        """Diagonal of the (normalized) selection Hamiltonian, cached on the instance.

        The selection Hamiltonian is built from the moments and the fixed
        ``selection_weighting`` and does NOT depend on K
        (``construct_selection_hubo_bin`` only stores ``selection_K``), so its
        diagonal is the same across the whole K-sweep and computed once. Under the
        default budget-weighting entry ``z`` is cost(n .* z) with n_i =
        budget/price_i; under unit-weighting it is the price-blind unit-weight
        energy -- both in the normalized units of ``final_expectation_value``.
        Diagonal of a diagonal operator, so dense over 2^N <= 32768.
        """
        diag = getattr(self, "_selection_diagonal", None)
        if diag is None:
            assert getattr(self, "selection_hamiltonian", None) is not None, \
                "construct_selection_hubo_bin must run before reading the diagonal"
            sparse = self.selection_hamiltonian.sparse_matrix(wire_order=range(self.num_assets))
            diag = np.real(np.asarray(sparse.diagonal())).astype(float)
            self._selection_diagonal = diag
        return diag

    def _approximation_ratios(self, K, final_expectation_value):
        """Subspace and global approximation ratios for the given K.

        Two ratios with deliberately *opposite* conventions (see report caveat):
        - subspace: standard form (final - C_random)/(C_opt - C_random) with
          C_random = Dicke-state mean, C_opt = subspace ground energy. 1 = optimal.
        - global: HOPO-parity min-max (final - E_min)/(E_max - E_min) over all
          2^N states, matching profile.py's integer-HUBO row. 0 = optimal.
        Raw energies are returned alongside so either convention is recomputable.

        IMPORTANT (TASK-03): these are *energy-convergence* ratios over the
        selection objective. Under the default budget-weighting the objective is
        cost(n .* y), so at K=1 the energy-min subset IS the best single-asset
        portfolio (the id-1 inversion below is fixed at the source); but at K>=2
        budget-weighting is only a proxy (the allocator distributes non-equally),
        so a high ratio still does not certify post-allocation quality. Under the
        legacy unit-weighting the gap is worse -- energy and post_objective are not
        co-monotone at all: e.g. id 1, K=1 selects NKE with
        approximation_ratio_subspace ~ 0.9999 yet post_objective ~ -136,281 (the
        worst single asset), while the best single-asset portfolio (TRV, ~ -2,106)
        is unreachable because unit-weight energy ranks NKE above TRV. Either way,
        judge portfolio quality by `post_objective` (see `solve_selection_exactly`),
        never by these ratios.
        """
        N = self.num_assets
        d = self._selection_energy_diagonal()
        sub = d[hamming_weight_indices(N, K)]

        E_sub_min = float(sub.min())
        E_sub_max = float(sub.max())
        E_dicke_mean = float(sub.mean())  # = <D_n^k| H |D_n^k>, uniform Dicke amplitudes
        E_glob_min = float(d.min())
        E_glob_max = float(d.max())

        final = float(final_expectation_value)
        sub_denom = E_sub_min - E_dicke_mean
        glob_denom = E_glob_max - E_glob_min
        # Degenerate K=N (single-state subspace) makes sub_denom ~ 0 -> None.
        ar_subspace = (final - E_dicke_mean) / sub_denom if abs(sub_denom) > 1e-12 else None
        ar_global = (final - E_glob_min) / glob_denom if abs(glob_denom) > 1e-12 else None

        return {
            "approximation_ratio_subspace": ar_subspace,
            "approximation_ratio_global": ar_global,
            "selection_energy_min_subspace": E_sub_min,
            "selection_energy_max_subspace": E_sub_max,
            "selection_energy_dicke_mean": E_dicke_mean,
            "selection_energy_global_min": E_glob_min,
            "selection_energy_global_max": E_glob_max,
        }

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

            if sum(weights.values()) < ZERO_WEIGHT_EPS:
                # The optimizer's answer is the empty portfolio; record it
                # deterministically instead of letting the tie-degenerate LP pick
                # an arbitrary allocation (see ZERO_WEIGHT_EPS). Empty portfolios
                # are not admissible benchmark answers (the K=0 exclusion), so
                # this is infeasible, mirroring min_unit_cost_exceeds_budget.
                return {
                    "allocation": {self.stocks[i]: 0 for i in range(self.num_assets)},
                    "realized_budget": 0.0,
                    "leftover_budget": float(self.budget),
                    "post_objective": None,
                    "infeasible": True,
                    "infeasible_reason": "zero_weight_portfolio",
                }

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

        effective_K = sum(1 for v in full_alloc.values() if v != 0)
        if effective_K == 0:
            # The LP bought nothing (another arbitrary vertex of the same tie,
            # or solver inaccuracy). A zero-share "portfolio" must not enter the
            # best_K / ranked_K comparison as a feasible result.
            return {
                "allocation": full_alloc,
                "realized_budget": 0.0,
                "leftover_budget": float(self.budget),
                "post_objective": None,
                "infeasible": True,
                "infeasible_reason": "empty_allocation",
            }

        post_obj = float(self.get_objective_value(dict(full_alloc)))

        return {
            "allocation": full_alloc,
            "realized_budget": float(realized),
            "leftover_budget": float(leftover),
            "post_objective": post_obj,
            "infeasible": False,
            "effective_K": effective_K,
        }

    # ------------------------------------------------------------------ #
    # Classical K-selector (replaces the brute-force K sweep).
    #
    # Stage 1: an L1-penalized continuous higher-moment solve localizes a
    # ballpark cardinality K_hat and ranks assets by |weight|.
    # Stage 2: a small explicit neighborhood of K around K_hat is scored with
    # the existing weighted higher-moment objective + discrete allocator
    # (`_allocate_on_subset`). No QAOA, no subset enumeration -> scales to large
    # N. The expensive fixed-K QAOA then runs only on the top-ranked K(s).
    # ------------------------------------------------------------------ #
    def lasso_localize_K(self, l1_lambdas=None, support_tol=1e-4,
                         target_lo=0.1, target_hi=0.9, k_min=1):
        """Localize a ballpark cardinality K_hat and an asset ranking via a
        short L1-penalty path. Returns ``(k_hat, ranking, support_path)`` where
        ``ranking`` is a list of asset indices ordered by descending |weight|
        and ``support_path`` records the support size at each lambda.

        This is a rough localizer, not an exact oracle: the higher-moment
        objective is non-convex, so the L1 solve only narrows K from [1, N] to a
        small neighborhood for the explicit scan in ``select_K_classically``.
        """
        N = self.num_assets
        if l1_lambdas is None:
            # Short geometric path, scaled by the expected-return magnitude so
            # the penalty bites across problems of different scale.
            scale = float(np.mean(np.abs(self.expected_returns))) or 1.0
            l1_lambdas = [c * scale for c in (1e-3, 1e-2, 1e-1, 1.0, 10.0)]

        have_moments = (self.coskewness_tensor is not None
                        and self.cokurtosis_tensor is not None)
        support_path = []
        solutions = []  # (lambda, weights_array, support_size)
        for lmbda in l1_lambdas:
            try:
                if have_moments:
                    hef = HigherMomentPortfolioOptimizer(
                        self.stocks, self.expected_returns, self.covariance_matrix,
                        self.coskewness_tensor, self.cokurtosis_tensor,
                        risk_aversion=self.risk_aversion)
                    _, w = hef.optimize_portfolio_with_higher_moments_l1(lmbda)
                else:
                    ef = EfficientFrontier(self.expected_returns, self.covariance_matrix)
                    ef.add_objective(objective_functions.L1_reg, gamma=lmbda)
                    w_idx = ef.max_quadratic_utility(risk_aversion=self.risk_aversion)
                    w = np.array([float(w_idx[i]) for i in range(N)])
            except Exception as e:
                support_path.append({"lambda": float(lmbda), "support": None, "error": str(e)})
                continue
            w = np.asarray(w, dtype=float)
            wmax = float(np.max(np.abs(w))) if w.size else 0.0
            thr = support_tol * wmax
            support = int(np.sum(np.abs(w) > thr)) if wmax > 0 else 0
            support_path.append({"lambda": float(lmbda), "support": support})
            solutions.append((float(lmbda), w, support))

        if not solutions:
            # Total failure across the path: fall back to a uniform ranking and a
            # mid-range K_hat; the explicit neighborhood scan still does the work.
            ranking = list(range(N))
            k_hat = int(min(max(k_min, round(N / 2)), N))
            return k_hat, ranking, support_path

        # Prefer the sparsest solution whose support fraction lands in the target
        # band; otherwise pick the support closest to the band midpoint (robust
        # to the all-zero / all-dense degenerate cases).
        in_range = [s for s in solutions
                    if target_lo * N <= s[2] <= target_hi * N and s[2] > 0]
        if in_range:
            chosen = min(in_range, key=lambda s: s[2])  # sparsest in range
        else:
            target_mid = 0.5 * (target_lo + target_hi) * N
            nonzero = [s for s in solutions if s[2] > 0]
            pool = nonzero if nonzero else solutions
            chosen = min(pool, key=lambda s: abs(s[2] - target_mid))

        _, w_chosen, support_chosen = chosen
        ranking = [int(i) for i in np.argsort(-np.abs(w_chosen))]
        if support_chosen > 0:
            k_hat = support_chosen
        else:
            k_hat = int(round(N / 2))
        k_hat = int(min(max(k_min, k_hat), N))
        return k_hat, ranking, support_path

    def select_K_classically(self, neighborhood=2, l1_lambdas=None, k_min=1):
        """Pick the best cardinality K classically (no QAOA).

        Localizes K_hat via ``lasso_localize_K`` then scores an explicit
        neighborhood ``[K_hat-neighborhood, K_hat+neighborhood]`` (clamped to
        ``[k_min, N]``) by allocating the top-K assets (by |LASSO weight|) with
        ``_allocate_on_subset`` and reading its fully-weighted ``post_objective``.
        Returns ``{k_hat, scan, ranked_K, support_path}`` with ``ranked_K`` the
        feasible candidate K sorted by classical objective (best first); exact
        score ties are broken toward the K closest to the winning allocation's
        ``effective_K`` (number of names actually held), then toward smaller K.

        Ranking is by allocated ``post_objective`` (not the unit-weight selection
        energy) by design: the two are not co-monotone, so the energy proxy is
        unreliable for portfolio quality (TASK-03; see ``_approximation_ratios``).
        """
        N = self.num_assets
        k_hat, ranking, support_path = self.lasso_localize_K(
            l1_lambdas=l1_lambdas, k_min=k_min)
        lo = max(k_min, k_hat - neighborhood)
        hi = min(N, k_hat + neighborhood)

        scan = []
        seen = set()

        def score_K(K):
            top_idx = sorted(ranking[:K])
            alloc = self._allocate_on_subset(top_idx)
            return {
                "K": K,
                "selected_indices": [int(i) for i in top_idx],
                "classical_post_objective": alloc["post_objective"],
                "infeasible": alloc["infeasible"],
                "infeasible_reason": alloc.get("infeasible_reason"),
                "effective_K": alloc.get("effective_K"),
            }

        for K in range(lo, hi + 1):
            scan.append(score_K(K))
            seen.add(K)

        feasible = [s for s in scan
                    if not s["infeasible"] and s["classical_post_objective"] is not None]

        # If nothing in the neighborhood is feasible, widen downward: a smaller K
        # means a cheaper subset, more likely to fit the budget.
        K_try = lo - 1
        while not feasible and K_try >= k_min:
            if K_try not in seen:
                s = score_K(K_try)
                scan.append(s)
                seen.add(K_try)
                if not s["infeasible"] and s["classical_post_objective"] is not None:
                    feasible.append(s)
            K_try -= 1

        # The classical score is computed on the top-K *superset* (the allocator
        # may hold fewer than K names), so exact ties across the window are
        # common. Among ties, prefer the K closest to the cardinality the
        # winning allocation actually holds — the exactly-K QAOA is most likely
        # to realize the classical score at K=effective_K — then smaller K.
        # (Previously ties fell back to scan order via sort stability.)
        ranked = sorted(feasible, key=lambda s: (
            -s["classical_post_objective"], abs(s["K"] - s["effective_K"]), s["K"]))
        scan.sort(key=lambda s: s["K"])
        return {
            "k_hat": int(k_hat),
            "scan": scan,
            "ranked_K": [s["K"] for s in ranked],
            "support_path": support_path,
        }

    def solve_selection_exactly(self, k_min=1, enumerate_post_objective=True,
                                enumeration_max_n=12):
        """Exact reference for the cardinality-selection problem (no QAOA).

        Symmetric to ``HigherOrderPortfolioQAOA.solve_exactly`` but for the
        ring-XY formulation. The selection Hamiltonian is K-independent, so its
        full diagonal over 2^N states (``_selection_energy_diagonal``) is the
        exact spectrum and is cheap (one qubit per asset, N small). Returns two
        first-class references:

        - ``energy_optimal``: per K the energy-minimizing selection (the true
          subspace ground state — what the QAOA tries to find), then the
          existing classical allocator (``_allocate_on_subset``) on that subset.
          ``approximation_ratio_subspace`` is 1.0 by construction (it *is* the
          subspace minimum). ``best_K``/``reference`` pick the feasible per-K
          entry with the best ``post_objective``. The energy bound
          ``selection_energy <= QAOA final_expectation_value`` is rigorous; the
          ``post_objective`` is a downstream quantity and is *not* guaranteed to
          dominate the QAOA's. Under the default budget-weighting this reflects
          cost(n .* y), so at K=1 ``energy_optimal.reference`` coincides with
          ``post_objective_optimal.reference`` (both pick the best single asset);
          at K>=2 they can still differ -- energy/post_objective co-monotonicity
          holds only at K=1.
        - ``post_objective_optimal`` (full subset enumeration over all
          C(N,K), K>=k_min): the genuine best ``post_objective`` the hybrid
          pipeline can reach. Because the QAOA's chosen subset at every K is one
          of these subsets and is scored by the same ``_allocate_on_subset``,
          ``post_objective_optimal.reference.post_objective`` is a guaranteed
          upper bound on every QAOA per-K ``post_objective``. Gated by
          ``enumeration_max_n`` (2^N allocations).

        The empty/K=0 exclusion and all infeasibility rules are inherited from
        ``_allocate_on_subset``; an infeasible/empty selection can never win
        ``best_K`` or the ``reference``.
        """
        N = self.num_assets
        k_min = max(1, min(int(k_min), N))

        # Cost guard: the diagonal of a 2^N x 2^N operator is impractical to
        # materialize for very large N. Never triggers on this dataset (N<=~10).
        spectrum_is_partial = N > 22
        if spectrum_is_partial:
            return {
                "k_min": k_min,
                "num_assets": int(N),
                "spectrum_is_partial": True,
                "selection_energy_global_min": None,
                "selection_energy_global_max": None,
                "global_argmin": None,
                "energy_optimal": {"per_K": [], "best_K": None, "reference": None},
                "post_objective_optimal": None,
            }

        # Build the (K-independent) Hamiltonian once and read its exact diagonal.
        self.construct_selection_hubo_bin(k_min)
        d = self._selection_energy_diagonal()

        def decode(z):
            bitstring = format(int(z), f"0{N}b")
            sel_idx = [q for q in range(N) if bitstring[q] == "1"]
            sel_stocks = [str(self.stocks[q]) for q in sel_idx]
            return bitstring, sel_idx, sel_stocks

        g_min = float(d.min())
        g_max = float(d.max())
        z_glob = int(d.argmin())
        gb, gi, gs = decode(z_glob)
        global_argmin = {
            "state_int": z_glob,
            "selection_bitstring": gb,
            "hamming_weight": int(bin(z_glob).count("1")),
            "selected_indices": [int(i) for i in gi],
            "selected_stocks": gs,
            "selection_energy": g_min,
        }

        # --- Energy-optimal reference: per-K subspace ground state + allocation.
        energy_per_K = []
        for K in range(k_min, N + 1):
            idxs = hamming_weight_indices(N, K)
            sub = d[idxs]
            z_K = int(idxs[int(sub.argmin())])
            E_sub_min = float(sub.min())
            bitstring, sel_idx, sel_stocks = decode(z_K)
            alloc = self._allocate_on_subset(sel_idx)
            ratios = self._approximation_ratios(K, E_sub_min)
            energy_per_K.append({
                "K": int(K),
                "selection_energy": E_sub_min,
                "selection_bitstring": bitstring,
                "selected_indices": [int(i) for i in sel_idx],
                "selected_stocks": sel_stocks,
                **ratios,
                "allocation": {str(s): int(v) for s, v in alloc["allocation"].items()},
                "realized_budget": float(alloc["realized_budget"]),
                "leftover_budget": float(alloc["leftover_budget"]),
                "post_objective": alloc["post_objective"],
                "infeasible": bool(alloc["infeasible"]),
                "infeasible_reason": alloc.get("infeasible_reason"),
                "effective_K": alloc.get("effective_K"),
            })

        feasible = [e for e in energy_per_K
                    if not e["infeasible"] and e["post_objective"] is not None]
        if feasible:
            best = sorted(feasible, key=lambda e: (
                -e["post_objective"], abs(e["K"] - e["effective_K"]), e["K"]))[0]
            energy_best_K = int(best["K"])
            energy_reference = dict(best)
        else:
            energy_best_K = None
            energy_reference = None

        energy_optimal = {
            "per_K": energy_per_K,
            "best_K": energy_best_K,
            "reference": energy_reference,
        }

        # --- Post-objective-optimal reference: best feasible subset over all
        # C(N,K) (K>=k_min). Guaranteed >= every QAOA per-K post_objective.
        post_objective_optimal = None
        if enumerate_post_objective and N <= int(enumeration_max_n):
            per_K_best = {}
            per_K_counts = {}  # K -> [n_subsets, n_feasible_subsets]
            global_best = None
            for z in range(2 ** N):
                K = bin(z).count("1")
                if K < k_min:
                    continue
                counts = per_K_counts.setdefault(K, [0, 0])
                counts[0] += 1
                bitstring, sel_idx, sel_stocks = decode(z)
                alloc = self._allocate_on_subset(sel_idx)
                if alloc["infeasible"] or alloc["post_objective"] is None:
                    continue
                counts[1] += 1
                entry = {
                    "K": int(K),
                    "selection_bitstring": bitstring,
                    "selected_indices": [int(i) for i in sel_idx],
                    "selected_stocks": sel_stocks,
                    "selection_energy": float(d[z]),
                    "allocation": {str(s): int(v) for s, v in alloc["allocation"].items()},
                    "realized_budget": float(alloc["realized_budget"]),
                    "leftover_budget": float(alloc["leftover_budget"]),
                    "post_objective": float(alloc["post_objective"]),
                    "infeasible": False,
                    "effective_K": alloc.get("effective_K"),
                }
                prev = per_K_best.get(K)
                if prev is None or entry["post_objective"] > prev["post_objective"]:
                    per_K_best[K] = entry
                if global_best is None or entry["post_objective"] > global_best["post_objective"]:
                    global_best = entry

            post_per_K = []
            for K in range(k_min, N + 1):
                counts = per_K_counts.get(K, [0, 0])
                best_K_entry = per_K_best.get(K)
                if best_K_entry is not None:
                    e = dict(best_K_entry)
                    e["n_subsets"] = int(counts[0])
                    e["n_feasible_subsets"] = int(counts[1])
                else:
                    e = {
                        "K": int(K),
                        "selection_bitstring": None,
                        "selected_indices": [],
                        "selected_stocks": [],
                        "selection_energy": None,
                        "allocation": {},
                        "realized_budget": 0.0,
                        "leftover_budget": float(self.budget),
                        "post_objective": None,
                        "infeasible": True,
                        "effective_K": None,
                        "n_subsets": int(counts[0]),
                        "n_feasible_subsets": 0,
                    }
                post_per_K.append(e)

            post_objective_optimal = {
                "enumerated": True,
                "per_K": post_per_K,
                "best_K": (int(global_best["K"]) if global_best is not None else None),
                "reference": (dict(global_best) if global_best is not None else None),
            }

        return {
            "k_min": k_min,
            "num_assets": int(N),
            "spectrum_is_partial": False,
            "selection_energy_global_min": g_min,
            "selection_energy_global_max": g_max,
            "global_argmin": global_argmin,
            "energy_optimal": energy_optimal,
            "post_objective_optimal": post_objective_optimal,
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
        """Fixed-K ring-XY QAOA selection, then a classical allocation on the
        chosen K-subset. Returns the selection, the `approximation_ratio_*`
        energy-convergence metrics, and the allocated `post_objective`.

        NOTE (TASK-03): the returned `approximation_ratio_subspace`/`_global`
        measure how well the QAOA minimized the *unit-weight* selection energy,
        NOT portfolio quality -- the two are not co-monotone (see
        `_approximation_ratios`). Downstream choice of K (best_K in experiments.py,
        ranked_K in `select_K_classically`) relies on `post_objective`, not these
        ratios.
        """
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
        approximation_ratios = self._approximation_ratios(K, final_expectation_value)

        return {
            "K": K,
            "layers": sel_layers,
            "final_expectation_value": final_expectation_value,
            **approximation_ratios,
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
            "effective_K": allocation_info.get("effective_K"),
        }
