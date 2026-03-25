import itertools
import time
import cma
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import PauliEvolutionGate
from pypfopt import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation
from scipy.optimize import minimize, approx_fprime

from portfolio_higher_moments_classical import HigherMomentPortfolioOptimizer
from portfolio_utils import basis_vector_to_bitstring, bitstrings_to_optimized_portfolios, int_to_bitstring, normalize_hamiltonian, smallest_eigenpairs, smallest_sparse_eigenpairs

np.random.seed(0)

class HigherOrderPortfolioQAOA:

    def __init__(self,
                 stocks,
                 prices_now,
                 expected_returns,
                 covariance_matrix,
                 budget,
                 max_qubits,
                 layers = None,
                 coskewness_tensor = None,
                 cokurtosis_tensor = None,
                 risk_aversion = 3,
                 mixer = "x",
                 log_encoding = False,
                 strict_budget_constraint = False,
                 lambda_budget = 1):
        # Implementation assumes that stocks and other data are ordered to match
        self.stocks = stocks
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.coskewness_tensor = coskewness_tensor
        self.cokurtosis_tensor = cokurtosis_tensor
        self.risk_aversion = risk_aversion
        self.mixer = mixer
        self.log_encoding = log_encoding
        self.budget = budget
        self.num_assets = len(expected_returns)
        self.prices_now = prices_now
        self.num_qubits_per_asset = {}
        self.strict_budget_constraint = strict_budget_constraint

        if log_encoding:
            # Smallest N such that 2^N > budget
            # Each asset can be bought at most floor(bugdet/price_now) times
            # Thus, for each asset we have to choose the smallest N such that 2^N > floor(bugdet/price_now)
            for asset in stocks:
                N = int(np.ceil(np.log2(np.floor(budget/prices_now[asset]))))
                if N == 0:
                    N = 1
                self.num_qubits_per_asset[asset] = N
                print(f"Number of qubits for asset {asset}: {N}")
        else:
            for asset in stocks:
                N = int(np.ceil(budget/prices_now[asset]))
                self.num_qubits_per_asset[asset] = N
                print(f"Number of qubits for asset {asset}: {N}")

        assert len(stocks) == len(expected_returns)
        assert len(expected_returns) == len(covariance_matrix)
        assert len(expected_returns) == len(covariance_matrix[0])
        assert risk_aversion > 0


        self.cost_hubo_int = {}
        self.cost_hubo_bin = {}
        self.assets_to_qubits = {}
        self.qubits_to_assets = {}

        self.n_qubits = 0
        for asset in stocks:
            self.assets_to_qubits[asset] = list(range(self.n_qubits, self.n_qubits + self.num_qubits_per_asset[asset]))
            for qubit in self.assets_to_qubits[asset]:
                self.qubits_to_assets[qubit] = asset
            self.n_qubits += self.num_qubits_per_asset[asset]

        self.layers = self.n_qubits if layers == None else layers
        self.layers = min(10, self.layers)
        self.init_params = 0.01*np.random.rand(2, self.layers)

        self.construct_cost_hubo_int()

        if strict_budget_constraint:
            self.budget_constraint = self.construct_budget_constraint_strict(scaler=lambda_budget)
        else:
            self.budget_constraint = self.construct_budget_constraint(scaler=lambda_budget)

        print("Total number of qubits: ", self.n_qubits)
        assert max_qubits >= self.n_qubits, "Number of qubits exceeds the maximum number of qubits"

        for var, coeff in self.budget_constraint.items():
            if var in self.cost_hubo_int:
                self.cost_hubo_int[var] += coeff
            else:
                self.cost_hubo_int[var] = coeff
        self.replace_integer_variables_with_binary_variables()
        self.simplify_cost_hubo_bin()
        self.cost_hubo_bin_to_ising_hamiltonian()
        self.qaoa_circuit, self.qaoa_probs_circuit = self.get_QAOA_circuits()


    def get_n_qubits(self):
        return self.n_qubits

    def get_layers(self):
        return self.layers

    def get_init_params(self):
        return self.init_params

    def construct_cost_hubo_int(self):
        for i in range(self.num_assets):
            mu = self.expected_returns[i]
            self.cost_hubo_int[(self.stocks[i],)] = -mu

        for i in range(self.num_assets):
            for j in range(self.num_assets):
                cov = (self.risk_aversion/2)*self.covariance_matrix[i][j]
                self.cost_hubo_int[(self.stocks[i], self.stocks[j])] = cov

        if self.coskewness_tensor is not None:
            for i in range(self.num_assets):
                for j in range(self.num_assets):
                    for k in range(self.num_assets):
                        skew = (self.risk_aversion/6)*self.coskewness_tensor[i][j][k]
                        self.cost_hubo_int[(self.stocks[i], self.stocks[j], self.stocks[k])] = -skew

        if self.cokurtosis_tensor is not None:
            for i in range(self.num_assets):
                for j in range(self.num_assets):
                    for k in range(self.num_assets):
                        for l in range(self.num_assets):
                            kurt = (self.risk_aversion/24)*self.risk_aversion*self.cokurtosis_tensor[i][j][k][l]
                            self.cost_hubo_int[(self.stocks[i], self.stocks[j], self.stocks[k], self.stocks[l])] = kurt


    def construct_budget_constraint(self, scaler = 1):
        budget_const = {}
        for asset in self.stocks:
            budget_const[(asset,)] = -2*self.budget*scaler*self.prices_now[asset]
            budget_const[(asset, asset)] = scaler*self.prices_now[asset]**2
        for asset1, asset2 in itertools.combinations(self.stocks, 2):
            budget_const[(asset1, asset2)] = 2*scaler*self.prices_now[asset1]*self.prices_now[asset2]
        return budget_const

    def construct_budget_constraint_strict(self, scaler = 1):
        scaler = self.budget**2
        from numpy import gcd
        self.budget_to_qubits = {}
        asset_prices_now = [int(self.prices_now[asset]) for asset in self.stocks]
        self.budget_dicretization_unit = gcd.reduce(asset_prices_now + [int(self.budget)])
        print("Budget discretization unit: ", self.budget_dicretization_unit)
        qubits_for_budget = int(np.floor(np.log2(self.budget/self.budget_dicretization_unit)))
        last_budget_qubit = self.n_qubits + qubits_for_budget
        budget_for_last_qubit = self.budget - self.budget_dicretization_unit*2**qubits_for_budget
        slacks = []
        self.assets_to_qubits[f"slack_int"] = list(range(self.n_qubits, self.n_qubits + qubits_for_budget))
        slacks.append(f"slack_int")
        self.n_qubits += qubits_for_budget
        self.assets_to_qubits[f"slack_int_last"] = [last_budget_qubit]
        slacks.append(f"slack_int_last")
        self.n_qubits += 1

        budget_qubit_coeffs = {"slack_int": self.budget_dicretization_unit, "slack_int_last": budget_for_last_qubit}

        all_vars = list(self.stocks) + slacks

        budget_const = {}
        for v in all_vars:
            if v in self.stocks:
                budget_const[(v, v)] = scaler*self.prices_now[v]**2
            else:
                budget_const[(v, v)] = scaler*budget_qubit_coeffs[v]**2

        for v1, v2 in itertools.combinations(all_vars, 2):
            if v1 in self.stocks and v2 in self.stocks:
                budget_const[(v1, v2)] = 2*scaler*self.prices_now[v1]*self.prices_now[v2]
            elif v1 in slacks and v2 in slacks:
                budget_const[(v1, v2)] = 2*scaler*budget_qubit_coeffs[v1]*budget_qubit_coeffs[v2]
            elif v1 in self.stocks and v2 in slacks:
                budget_const[(v1, v2)] = -2*scaler*self.prices_now[v1]*budget_qubit_coeffs[v2]
            elif v1 in slacks and v2 in self.stocks:
                budget_const[(v1, v2)] = -2*scaler*self.prices_now[v2]*budget_qubit_coeffs[v1]

        return budget_const

    def replace_integer_variables_with_binary_variables(self):
        """
        Every integer variable m, labeled with stocks, is replaced with binary variables
        meaning integer variable m is replaced with a sum of binary variables x_0 + 2*x_1 + 4*x_2 + 8*x_3 + ... + 2^(n)*x_{n}
        where n = ceil(log2(budget)) and x_n are binary.
        Then, for integer variables m and k, we have m*k = sum_{i=0}^{n_m} sum_{j=0}^{n_k} 2^{i+j} x_i x_j, etc.
        """
        for integer_var in self.cost_hubo_int:
            if len(integer_var) == 1:
                asset = integer_var[0]
                for n, qubit in enumerate(self.assets_to_qubits[asset]):
                    bin_var = ((asset, qubit),)
                    if self.log_encoding:
                        if bin_var in self.cost_hubo_bin:
                            self.cost_hubo_bin[bin_var] += (2**n)*self.cost_hubo_int[integer_var]
                        else:
                            self.cost_hubo_bin[bin_var] = (2**n)*self.cost_hubo_int[integer_var]
                    else:
                        self.cost_hubo_bin[bin_var] = self.cost_hubo_int[integer_var]
            elif len(integer_var) == 2:
                asset0 = integer_var[0]
                asset1 = integer_var[1]
                for n, qubit0 in enumerate(self.assets_to_qubits[asset0]):
                    for m, qubit1 in enumerate(self.assets_to_qubits[asset1]):
                        bin_var = ((asset0, qubit0), (asset1, qubit1))
                        if self.log_encoding:
                            if bin_var in self.cost_hubo_bin:
                                self.cost_hubo_bin[bin_var] += 2**(n+m)*self.cost_hubo_int[integer_var]
                            else:
                                self.cost_hubo_bin[bin_var] = 2**(n+m)*self.cost_hubo_int[integer_var]
                        else:
                            self.cost_hubo_bin[bin_var] = self.cost_hubo_int[integer_var]
            elif len(integer_var) == 3:
                asset0 = integer_var[0]
                asset1 = integer_var[1]
                asset2 = integer_var[2]
                for n, qubit0 in enumerate(self.assets_to_qubits[asset0]):
                    for m, qubit1 in enumerate(self.assets_to_qubits[asset1]):
                        for k, qubit2 in enumerate(self.assets_to_qubits[asset2]):
                            bin_var = ((asset0, qubit0), (asset1, qubit1), (asset2, qubit2))
                            if self.log_encoding:
                                self.cost_hubo_bin[bin_var] = 2**(n+m+k)*self.cost_hubo_int[integer_var]
                            else:
                                self.cost_hubo_bin[bin_var] = self.cost_hubo_int[integer_var]
            elif len(integer_var) == 4:
                asset0 = integer_var[0]
                asset1 = integer_var[1]
                asset2 = integer_var[2]
                asset3 = integer_var[3]
                for n, qubit0 in enumerate(self.assets_to_qubits[asset0]):
                    for m, qubit1 in enumerate(self.assets_to_qubits[asset1]):
                        for k, qubit2 in enumerate(self.assets_to_qubits[asset2]):
                            for l, qubit3 in enumerate(self.assets_to_qubits[asset3]):
                                bin_var = ((asset0, qubit0), (asset1, qubit1), (asset2, qubit2), (asset3, qubit3))
                                if self.log_encoding:
                                    self.cost_hubo_bin[bin_var] = 2**(n+m+k+l)*self.cost_hubo_int[integer_var]
                                else:
                                    self.cost_hubo_bin[bin_var] = self.cost_hubo_int[integer_var]

    def simplify_cost_hubo_bin(self):
        # Using the fact that x_i^n = x_i, for n > 0, we can simplify the cost function
        self.cost_hubo_bin_simplified = {}
        for bin_var in self.cost_hubo_bin:
            bin_var_set = frozenset(bin_var)
            if bin_var_set in self.cost_hubo_bin_simplified:
                self.cost_hubo_bin_simplified[bin_var_set] += self.cost_hubo_bin[bin_var]
            else:
                self.cost_hubo_bin_simplified[bin_var_set] = self.cost_hubo_bin[bin_var]


    def cost_hubo_bin_to_ising_hamiltonian(self):
        """
        Convert the binary cost function to an Ising Hamiltonian using SparsePauliOp.
        Performs the standard x = (1 - z)/2 substitution: each binary variable x_i maps to (I - Z_i)/2.
        Products of binary variables expand into sums of Pauli-Z strings.

        Qiskit uses little-endian qubit ordering: qubit 0 is the rightmost character in the Pauli string.
        """
        all_terms = []
        for bin_var_set in self.cost_hubo_bin_simplified:
            bin_var = list(bin_var_set)
            coeff = self.cost_hubo_bin_simplified[bin_var_set]
            qubit_indices = [var[1] for var in bin_var]
            k = len(qubit_indices)

            # Expand product_i (I_i - Z_i) into 2^k Pauli strings
            for signs in itertools.product([0, 1], repeat=k):
                label = ['I'] * self.n_qubits
                sign_coeff = 1
                for j, s in enumerate(signs):
                    if s == 1:
                        label[qubit_indices[j]] = 'Z'
                        sign_coeff *= -1
                # Reverse for Qiskit little-endian convention (qubit 0 = rightmost)
                pauli_str = ''.join(reversed(label))
                all_terms.append((pauli_str, coeff / (2 ** k) * sign_coeff))

        self.hamiltonian = SparsePauliOp.from_list(all_terms).simplify()


    def _build_qaoa_circuit(self, cost_hamiltonian, n_qubits, layers):
        """Build a parameterized QAOA QuantumCircuit."""
        # X-mixer Hamiltonian: sum_i X_i
        x_labels = []
        for i in range(n_qubits):
            label = ['I'] * n_qubits
            label[i] = 'X'
            # Reverse for Qiskit little-endian convention
            x_labels.append((''.join(reversed(label)), 1.0))
        mixer_op = SparsePauliOp.from_list(x_labels)

        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))

        gammas = [Parameter(f'gamma_{l}') for l in range(layers)]
        alphas = [Parameter(f'alpha_{l}') for l in range(layers)]

        for l in range(layers):
            # Cost layer: exp(-i * gamma * H_cost)
            # All Z terms commute, so Trotter decomposition is exact with reps=1
            cost_evo = PauliEvolutionGate(cost_hamiltonian, time=gammas[l])
            qc.append(cost_evo, range(n_qubits))
            # Mixer layer: exp(-i * alpha * H_mixer)
            mixer_evo = PauliEvolutionGate(mixer_op, time=alphas[l])
            qc.append(mixer_evo, range(n_qubits))

        return qc, gammas, alphas

    def _evaluate_qaoa(self, qc, gammas, alphas, params, cost_hamiltonian):
        """Compute expectation value <psi|H|psi> for given QAOA parameters."""
        gamma_vals = params[:len(gammas)]
        alpha_vals = params[len(gammas):]
        param_dict = dict(zip(gammas + alphas, np.concatenate([gamma_vals, alpha_vals])))
        bound = qc.assign_parameters(param_dict)
        sv = Statevector(bound)
        return float(sv.expectation_value(cost_hamiltonian).real)

    def _get_probs(self, qc, gammas, alphas, params):
        """Get probability distribution over computational basis states."""
        gamma_vals = params[:len(gammas)]
        alpha_vals = params[len(gammas):]
        param_dict = dict(zip(gammas + alphas, np.concatenate([gamma_vals, alpha_vals])))
        bound = qc.assign_parameters(param_dict)
        sv = Statevector(bound)
        return np.array(sv.probabilities())


    def get_QAOA_circuits(self):
        cost_hamiltonian = self.get_cost_hamiltonian()
        qc, gammas, alphas = self._build_qaoa_circuit(cost_hamiltonian, self.n_qubits, self.layers)

        def qaoa_circuit(params):
            return self._evaluate_qaoa(qc, gammas, alphas, params, cost_hamiltonian)

        def qaoa_probs_circuit(params):
            return self._get_probs(qc, gammas, alphas, params)

        return qaoa_circuit, qaoa_probs_circuit


    def draw_qaoa_circuit(self):
        cost_hamiltonian = self.get_cost_hamiltonian()
        qc, gammas, alphas = self._build_qaoa_circuit(cost_hamiltonian, self.n_qubits, self.layers)
        fig = qc.draw('mpl')
        fig.savefig("qaoa_circuit.png")


    def get_latex_qaoa_circuit(self):
        cost_hamiltonian = self.get_cost_hamiltonian()
        qc, gammas, alphas = self._build_qaoa_circuit(cost_hamiltonian, self.n_qubits, 1)
        # Bind random parameters for visualization
        param_vals = np.pi * np.random.rand(2)
        param_dict = dict(zip(gammas + alphas, [param_vals[0], param_vals[1]]))
        bound = qc.assign_parameters(param_dict)
        latex = bound.draw(output="latex_source")
        return latex


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


    def solve_exactly(self):
        if self.n_qubits < 14:
            cost_matrix = self.get_cost_hamiltonian().to_matrix()
            self.smallest_eigenvalues, self.smallest_eigenvectors, first_excited_energy, first_excited_state, eigenvalues = smallest_eigenpairs(cost_matrix)
        else:
            cost_matrix = self.get_cost_hamiltonian().to_matrix(sparse=True)
            self.smallest_eigenvalues, self.smallest_eigenvectors, eigenvalues = smallest_sparse_eigenpairs(cost_matrix)
            first_excited_energy = None
            first_excited_state = None

        self.smallest_bitstrings = [basis_vector_to_bitstring(v) for v in self.smallest_eigenvectors]
        optimized_portfolio = bitstrings_to_optimized_portfolios(self.smallest_bitstrings, self.assets_to_qubits)
        result1 = self.satisfy_budget_constraint(optimized_portfolio)
        eigenvalues = [float(v) for v in eigenvalues]
        objective_values = [float(self.get_objective_value(allocation)) for allocation in optimized_portfolio]
        for i, r in enumerate(result1):
            r["objective_value"] = objective_values[i]

        second_optimized_portfolio = None
        result2 = None
        if first_excited_state is not None:
            second_smallest_bitstrings = [basis_vector_to_bitstring(first_excited_state)]
            second_optimized_portfolio = bitstrings_to_optimized_portfolios(second_smallest_bitstrings, self.assets_to_qubits)
            result2 = self.satisfy_budget_constraint(second_optimized_portfolio)
            objective_values = [float(self.get_objective_value(allocation)) for allocation in second_optimized_portfolio]
            for i, r in enumerate(result2):
                r["objective_value"] = objective_values[i]

        return self.smallest_eigenvalues, self.smallest_bitstrings, first_excited_energy, optimized_portfolio, second_optimized_portfolio, eigenvalues, result1, result2


    def satisfy_budget_constraint(self, optimized_portfolio):
        # Satisfies budget constraint
        results = []
        for portfolio in optimized_portfolio:
            B = 0
            for asset in portfolio:
                if asset in self.stocks:
                    B += self.prices_now[asset]*portfolio[asset]
            if B > self.budget:
                print("Budget constraint not satisfied", B, self.budget, "Difference: ", self.budget - B)
            else:
                print("Budget constraint satisfied", B, self.budget, "Difference: ", self.budget - B)
            results.append({"portfolio": portfolio, "budget": B, "difference": self.budget - B})
        return results


    def solve_with_qaoa(self):
        params = self.init_params.copy()
        # Flatten init_params from (2, layers) to (2*layers,) for the circuit callables
        flat_params = np.concatenate((params[0], params[1]))
        probs = self.qaoa_probs_circuit(flat_params)
        total_steps = 500

        # Use scipy L-BFGS-B as replacement for PennyLane AdagradOptimizer
        def objective(p):
            return self.qaoa_circuit(p)

        result = minimize(objective, flat_params, method='L-BFGS-B',
                         options={'maxiter': total_steps})
        flat_params = result.x

        probs = self.qaoa_probs_circuit(flat_params)
        final_expectation_value = self.qaoa_circuit(flat_params)
        two_most_probable_states = np.argsort(probs)[-2:]
        states_probs = [probs[i] for i in two_most_probable_states]
        two_most_probable_states = [int_to_bitstring(i, self.n_qubits) for i in two_most_probable_states]
        optimized_portfolios = bitstrings_to_optimized_portfolios(two_most_probable_states, self.assets_to_qubits)

        self.satisfy_budget_constraint(optimized_portfolios)

        return two_most_probable_states, final_expectation_value, flat_params, total_steps, states_probs, optimized_portfolios


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


    def solve_with_qaoa_cma_es(self):
        maxiter = 800
        if self.n_qubits > 13:
            maxiter = 300

        cost_hamiltonian = self.get_cost_hamiltonian()
        qc, gammas, alphas = self._build_qaoa_circuit(cost_hamiltonian, self.n_qubits, self.layers)

        def objective_function(params):
            return self._evaluate_qaoa(qc, gammas, alphas, params, cost_hamiltonian)

        initial_params = np.pi*np.random.rand(2, self.layers)
        # Make initial params 1-D array
        initial_params = np.concatenate((initial_params[0], initial_params[1]))
        print("Initial params: ", initial_params)
        es = cma.CMAEvolutionStrategy(initial_params, sigma0=0.1, options={"maxiter": maxiter})
        result = es.optimize(objective_function)
        optimized_params = result.result.xbest
        final_expectation_value = self._evaluate_qaoa(qc, gammas, alphas, optimized_params, cost_hamiltonian)
        probs = self._get_probs(qc, gammas, alphas, optimized_params)
        two_most_probable = np.argsort(probs)[-2:]
        states_probs = [probs[i] for i in two_most_probable]
        two_most_probable_states = [int_to_bitstring(i, self.n_qubits) for i in two_most_probable]
        optimized_portfolios = bitstrings_to_optimized_portfolios(two_most_probable_states, self.assets_to_qubits)
        result1 = self.satisfy_budget_constraint(optimized_portfolios)
        objective_values = [float(self.get_objective_value(optimized_portfolios[i])) for i in range(2)]
        training_history = self.cma_result_to_dict(result.result)
        return (two_most_probable_states,
                final_expectation_value,
                optimized_params,
                es.result.iterations,
                states_probs,
                optimized_portfolios,
                training_history,
                objective_values,
                result1)

    def solve_with_qaoa_scipy(self, optimizer='COBYLA'):
        """
        Solve the optimization problem using QAOA with a specified SciPy optimizer.

        Parameters:
        -----------
        optimizer : str
            The name of the SciPy optimizer to use. Must be one of:
            'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B',
            'TNC', 'COBYLA', 'COBYQA', 'SLSQP', 'trust-constr', 'dogleg',
            'trust-ncg', 'trust-krylov', 'trust-exact'

        Returns:
        --------
        tuple: Results of the optimization including states, expectation value, parameters, etc.
        """
        # Validate the optimizer choice
        valid_optimizers = [
            'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B',
            'TNC', 'COBYLA', 'COBYQA', 'SLSQP', 'trust-constr', 'dogleg',
            'trust-ncg', 'trust-krylov', 'trust-exact'
        ]

        if optimizer not in valid_optimizers:
            raise ValueError(f"Optimizer must be one of {valid_optimizers}")

        # Set maximum iterations based on problem size
        maxiter = 10000
        if self.n_qubits > 13:
            maxiter = 2500

        # Build QAOA circuit
        cost_hamiltonian = self.get_cost_hamiltonian()
        qc, gammas, alphas = self._build_qaoa_circuit(cost_hamiltonian, self.n_qubits, self.layers)

        # Track function evaluations
        func_evals = [0]

        def objective_function(params):
            func_evals[0] += 1
            return self._evaluate_qaoa(qc, gammas, alphas, params, cost_hamiltonian)

        # Generate initial parameters
        initial_params = np.pi * np.random.rand(2, self.layers)
        initial_params = np.concatenate((initial_params[0], initial_params[1]))

        # Define optimization options
        options = {'maxiter': maxiter}

        # Some optimizers require bounds or additional parameters
        bounds = None
        if optimizer in ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr']:
            bounds = [(0, 2*np.pi) for _ in range(len(initial_params))]

        # Extra options for specific optimizers
        if optimizer in ['trust-constr', 'trust-ncg', 'trust-krylov', 'trust-exact']:
            options['gtol'] = 1e-5

        jac = None
        hess = None
        if optimizer in ['Newton-CG', 'trust-ncg', 'trust-krylov', 'trust-exact', 'dogleg']:
            jac = lambda x : approx_fprime(x, objective_function, 0.01)

            if optimizer in ['trust-exact', 'dogleg', 'trust-ncg', 'trust-krylov']:

                def hessian(x):
                    epsilon = np.sqrt(np.finfo(float).eps)
                    n = len(x)
                    H = np.zeros((n, n))

                    for i in range(n):
                        def partial_derivative_i(x):
                            return approx_fprime(x, objective_function, epsilon)[i]

                        H[i] = approx_fprime(x, partial_derivative_i, epsilon)

                    H = (H + H.T) / 2
                    return H

                hess = hessian

        # Define callback that prints some progress
        def callback(xk):
            if func_evals[0] % 100 == 0:
                print(f"Iteration: {func_evals[0]}")
                print(f"Objective: {objective_function(xk)}")

        # Run the optimization with the selected optimizer
        start_time = time.time()
        result = minimize(
            objective_function,
            initial_params,
            method=optimizer,
            bounds=bounds,
            options=options,
            jac=jac,
            hess=hess,
            callback=callback
        )
        end_time = time.time()

        # Get optimized results
        optimized_params = result.x
        final_expectation_value = self._evaluate_qaoa(qc, gammas, alphas, optimized_params, cost_hamiltonian)
        probs = self._get_probs(qc, gammas, alphas, optimized_params)

        # Extract the two most probable states
        two_most_probable = np.argsort(probs)[-2:]
        states_probs = [probs[i] for i in two_most_probable]
        two_most_probable_states = [int_to_bitstring(i, self.n_qubits) for i in two_most_probable]

        # Process results
        optimized_portfolios = bitstrings_to_optimized_portfolios(two_most_probable_states, self.assets_to_qubits)
        result1 = self.satisfy_budget_constraint(optimized_portfolios)
        objective_values = [float(self.get_objective_value(optimized_portfolios[i])) for i in range(2)]

        # Create training history
        iterations = getattr(result, 'nit', None)
        if iterations is None:
            iterations = func_evals[0]

        training_history = {
            'iterations': int(iterations) if iterations is not None else None,
            'evaluations': int(func_evals[0]),
            'final_value': float(final_expectation_value),
            'success': bool(result.success),
            'status': int(result.status),
            'message': str(result.message),
            'execution_time': float(end_time - start_time),
            'optimizer': str(optimizer)
        }

        return (
            two_most_probable_states,
            final_expectation_value,
            optimized_params,
            iterations,
            states_probs,
            optimized_portfolios,
            training_history,
            objective_values,
            result1
        )

    def get_assets_to_qubits(self):
        return self.assets_to_qubits

    def get_qubits_to_assets(self):
        return self.qubits_to_assets

    def get_cost_hubo_int(self):
        return self.cost_hubo_int

    def get_cost_hubo_bin(self):
        return self.cost_hubo_bin

    def get_cost_hubo_bin_simplified(self):
        return self.cost_hubo_bin_simplified

    def get_cost_hamiltonian(self, normalized = True):
        if normalized:
            return normalize_hamiltonian(self.hamiltonian)
        return self.hamiltonian


    def solve_with_iterative_QAOA(self, max_layers = 10):
        for layers in range(1, max_layers + 1):
            print(f"Trying with {layers} layers")
            self.layers = layers
            self.qaoa_circuit, self.qaoa_probs_circuit = self.get_QAOA_circuits()

            if layers == 1:
                params = np.concatenate([0.01*np.random.rand(layers), 0.01*np.random.rand(layers)])
            else:
                # Extend previous params by repeating last value
                old_gammas = params[:layers-1]
                old_alphas = params[layers-1:2*(layers-1)]
                params = np.concatenate([old_gammas, [old_gammas[-1]], old_alphas, [old_alphas[-1]]])

            # Use scipy L-BFGS-B as replacement for PennyLane AdagradOptimizer
            def objective(p):
                return self.qaoa_circuit(p)

            result = minimize(objective, params, method='L-BFGS-B',
                            options={'maxiter': 2000})
            params = result.x

            # Print the results
            probs = self.qaoa_probs_circuit(params)
            final_expectation_value = self.qaoa_circuit(params)
            two_most_probable_states = np.argsort(probs)[-2:]
            states_probs = [probs[i] for i in two_most_probable_states]
            two_most_probable_states = [int_to_bitstring(i, self.n_qubits) for i in two_most_probable_states]
            optimized_portfolios = bitstrings_to_optimized_portfolios(two_most_probable_states, self.assets_to_qubits)

            self.satisfy_budget_constraint(optimized_portfolios)

            print(f"Two most probable states: {two_most_probable_states} with probabilities {states_probs}")
            print(f"Final expectation value: {final_expectation_value}")
            print(f"Optimized portfolios: {optimized_portfolios}")
            objective_values = [self.get_objective_value(optimized_portfolios[i]) for i in range(2)]
            print(f"Objective values: {objective_values}")
            if "".join(two_most_probable_states[-1]) in self.smallest_bitstrings:
                break

        return two_most_probable_states, final_expectation_value, params, 2000, states_probs, optimized_portfolios


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

    def warm_start_qaoa(self, target_bitstring):
        params = self.init_params.copy()
        flat_params = np.concatenate((params[0], params[1]))
        total_steps = 200
        epsilon = 1e-8
        target_probs = np.full(2**self.n_qubits, epsilon)
        target_probs[int(target_bitstring, 2)] = 1 - (2**(self.n_qubits) - 1)*epsilon

        _, qaoa_probs_circuit = self.get_QAOA_circuits()

        def objective(p):
            probs = qaoa_probs_circuit(p)
            # KL divergence
            return np.sum(target_probs * (np.log2(target_probs) - np.log2(probs)))

        result = minimize(objective, flat_params, method='L-BFGS-B',
                         options={'maxiter': total_steps})
        return result.x

    def solve_exactly_with_lobpcg(self):
        from scipy.sparse.linalg import lobpcg

        k = 2
        n = 2**self.n_qubits
        rng = np.random.default_rng()
        X = rng.normal(size=(n, k))
        X = X.astype(np.float32)

        cost_matrix = self.get_cost_hamiltonian().to_matrix(sparse=True)

        eigenvalues, v = lobpcg(cost_matrix, X, largest=False, tol=1e-5)
        self.smallest_eigenvalues = [eigenvalues[0]]
        eigenvect = v[:, 0]
        argmax = np.argmax(np.abs(eigenvect))
        new_eigenvect = np.zeros_like(eigenvect)
        new_eigenvect[argmax] = 1
        self.smallest_eigenvectors = [new_eigenvect]
        first_excited_energy = [eigenvalues[1]]
        eigenvect = v[:, 1]
        argmax = np.argmax(np.abs(eigenvect))
        new_eigenvect = np.zeros_like(eigenvect)
        new_eigenvect[argmax] = 1
        first_excited_state = new_eigenvect

        print("Smallest eigenvalues: ", self.smallest_eigenvalues)
        print("First excited energy: ", first_excited_energy)
        print("Smallest eigenvectors: ", self.smallest_eigenvectors)
        print("First excited state: ", first_excited_state)

        self.smallest_bitstrings = [basis_vector_to_bitstring(v) for v in self.smallest_eigenvectors]
        print("Smallest bitstrings: ", self.smallest_bitstrings)
        print(self.assets_to_qubits)
        optimized_portfolio = bitstrings_to_optimized_portfolios(self.smallest_bitstrings, self.assets_to_qubits)
        result1 = self.satisfy_budget_constraint(optimized_portfolio)
        eigenvalues = [float(v) for v in eigenvalues]

        objective_values = [float(self.get_objective_value(allocation)) for allocation in optimized_portfolio]
        for i, r in enumerate(result1):
            r["objective_value"] = objective_values[i]

        second_optimized_portfolio = None
        result2 = None
        if first_excited_state is not None:
            second_smallest_bitstrings = [basis_vector_to_bitstring(first_excited_state)]
            second_optimized_portfolio = bitstrings_to_optimized_portfolios(second_smallest_bitstrings, self.assets_to_qubits)
            result2 = self.satisfy_budget_constraint(second_optimized_portfolio)
            objective_values = [float(self.get_objective_value(allocation)) for allocation in second_optimized_portfolio]
            for i, r in enumerate(result2):
                r["objective_value"] = objective_values[i]

        return (self.smallest_eigenvalues,
                self.smallest_bitstrings,
                first_excited_energy,
                optimized_portfolio,
                second_optimized_portfolio,
                eigenvalues,
                result1,
                result2)
