import itertools
import time
import cma
import pennylane as qml
from pennylane import numpy as np
from pypfopt import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation
from scipy.optimize import minimize, approx_fprime

from portfolio_higher_moments_classical import HigherMomentPortfolioOptimizer
from utils import basis_vector_to_bitstring, bitstrings_to_optimized_portfolios, int_to_bitstring, normalize_linear_combination, replace_h_rz_h_with_rx, smallest_eigenpairs, smallest_sparse_eigenpairs

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
        
        #print("Number of qubits per asset: ", self.num_qubits_per_asset)

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
        self.init_params = 0.01*np.random.rand(2, self.layers, requires_grad=True)
        
        #print("Constructing cost hubo with integer variables")
        self.construct_cost_hubo_int()

        if strict_budget_constraint:
            self.budget_constraint = self.construct_budget_constraint_strict(scaler=lambda_budget)
        else:
            self.budget_constraint = self.construct_budget_constraint(scaler=lambda_budget)

        print("Total number of qubits: ", self.n_qubits)
        assert max_qubits >= self.n_qubits, "Number of qubits exceeds the maximum number of qubits"
        
        #print("Adding budget constraints to the cost function -> constructing full hubo problem")
        for var, coeff in self.budget_constraint.items():
            if var in self.cost_hubo_int:
                self.cost_hubo_int[var] += coeff
            else:
                self.cost_hubo_int[var] = coeff
        #print("Replacing integer variables with binary variables")
        self.replace_integer_variables_with_binary_variables()
        #print("Simplifying the binary cost function")
        self.simplify_cost_hubo_bin()
        #print("Converting binary cost function to Ising Hamiltonian")
        self.cost_hubo_bin_to_ising_hamiltonian()
        #print("Constructing QAOA circuits")
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
        # Sum over all variables sums to budget (sum_{v in vars} v - budget in terms of assets)**2
        # Recall that the variables at this stage are integers so x^n != x
        # But in reality, not price of every stock is one, so in reality we have
        # (sum_{i=0}^{n} price_i z_i - budget in terms of dollars)**2 = 0
        budget_const = {}
        for asset in self.stocks:
            budget_const[(asset,)] = -2*self.budget*scaler*self.prices_now[asset]
            budget_const[(asset, asset)] = scaler*self.prices_now[asset]**2
        for asset1, asset2 in itertools.combinations(self.stocks, 2):
            budget_const[(asset1, asset2)] = 2*scaler*self.prices_now[asset1]*self.prices_now[asset2]
        return budget_const

    def construct_budget_constraint_strict(self, scaler = 1):
        scaler = self.budget**2 #/(len(self.stocks) * self.budget)
        from numpy import gcd
        # Sum over all variables sums to budget (sum_{v in vars} v - budget in terms of assets)**2
        # Recall that the variables at this stage are integers so x^n != x
        # But in reality, not price of every stock is one, so in reality we have
        # (sum_{i=0}^{n} price_i z_i - budget in terms of dollars)**2 = 0

        # Encode budget using integer slack variables for the assets
        # self.budget = sum of slack variables

        self.budget_to_qubits = {}
        asset_prices_now = [int(self.prices_now[asset]) for asset in self.stocks]
        self.budget_dicretization_unit = gcd.reduce(asset_prices_now + [int(self.budget)])
        print("Budget discretization unit: ", self.budget_dicretization_unit)
        qubits_for_budget = int(np.floor(np.log2(self.budget/self.budget_dicretization_unit)))
        last_budget_qubit = self.n_qubits + qubits_for_budget
        budget_for_last_qubit = self.budget - self.budget_dicretization_unit*2**qubits_for_budget
        slacks = []
        #for i in range(qubits_for_budget):
        self.assets_to_qubits[f"slack_int"] = list(range(self.n_qubits, self.n_qubits + qubits_for_budget))
        slacks.append(f"slack_int")
        self.n_qubits += qubits_for_budget
        self.assets_to_qubits[f"slack_int_last"] = [last_budget_qubit]
        slacks.append(f"slack_int_last")
        self.n_qubits += 1

        budget_qubit_coeffs = {"slack_int": self.budget_dicretization_unit, "slack_int_last": budget_for_last_qubit}

        # Integer variables are self.budget_dicretization_units

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
        # This function performs the standard x = (1 + z)/2 substitution to convert the binary cost function to an Ising Hamiltonian
        self.hamiltonian = 0
        for bin_var_set in self.cost_hubo_bin_simplified:
            bin_var = list(bin_var_set)
            coeff = self.cost_hubo_bin_simplified[bin_var_set]
            if len(bin_var) == 1:

                qubit0 = bin_var[0][1]
                self.hamiltonian += (coeff/2)*(qml.Identity(qubit0) - qml.PauliZ(qubit0))

            elif len(bin_var) == 2:

                qubit0 = bin_var[0][1]
                qubit1 = bin_var[1][1]
                self.hamiltonian += (coeff/4)*(qml.Identity(qubit0) - qml.PauliZ(qubit0)) @ (qml.Identity(qubit1) - qml.PauliZ(qubit1))

            elif len(bin_var) == 3:

                qubit0 = bin_var[0][1]
                qubit1 = bin_var[1][1]
                qubit2 = bin_var[2][1]
                
                self.hamiltonian += (coeff/8)*(qml.Identity(qubit0) - qml.PauliZ(qubit0)) @ (qml.Identity(qubit1) - qml.PauliZ(qubit1)) @ (qml.Identity(qubit2) - qml.PauliZ(qubit2))
                
            elif len(bin_var) == 4:

                qubit0 = bin_var[0][1]
                qubit1 = bin_var[1][1]
                qubit2 = bin_var[2][1]
                qubit3 = bin_var[3][1]
                
                self.hamiltonian += (coeff/16)*(qml.Identity(qubit0) - qml.PauliZ(qubit0)) @ (qml.Identity(qubit1) - qml.PauliZ(qubit1)) @ (qml.Identity(qubit2) - qml.PauliZ(qubit2)) @ (qml.Identity(qubit3) - qml.PauliZ(qubit3))

                #print("Hamiltonian: ", self.hamiltonian)
                

    def get_QAOA_circuits(self):
        dev = qml.device('default.qubit', wires=self.n_qubits)
        
        cost_hamiltonian = self.get_cost_hamiltonian()
        mixer_hamiltonian = qml.qaoa.x_mixer(range(self.n_qubits)) #qml.qaoa.xy_mixer(complete_graph)

        def qaoa_layer(gamma, alpha):
            qml.qaoa.cost_layer(gamma, cost_hamiltonian)
            qml.qaoa.mixer_layer(alpha, mixer_hamiltonian)
        
        @qml.qnode(dev)
        def qaoa_circuit(params):
            for wire in range(self.n_qubits):
                qml.Hadamard(wires=wire)
            qml.layer(qaoa_layer, self.layers, params[0], params[1])
            return qml.expval(cost_hamiltonian)
        
        @qml.qnode(dev)
        def qaoa_probs_circuit(params):
            for wire in range(self.n_qubits):
                qml.Hadamard(wires=wire)
            qml.layer(qaoa_layer, self.layers, params[0], params[1])
            return qml.probs()

        return qaoa_circuit, qaoa_probs_circuit
    

    def get_compiled_QAOA_circuits(self):
        # Compile the QAOA circuit to some specific gate set
        # It seems that Pennylane compiler is too eager to decompose, 
        # since it unnecessarily applies the rule RX = H RZ H
        qaoa_circuit, qaoa_probs_circuit = self.get_QAOA_circuits()
        allowed_gates = ["CNOT", "RZ", "RX", "Hadamard"]
        dispatched_transform = qml.transform(replace_h_rz_h_with_rx)
        qaoa_circuit = qml.compile(qaoa_circuit, basis_set = allowed_gates)
        qaoa_circuit = qml.compile(qaoa_circuit, pipeline = [dispatched_transform])

        qaoa_probs_circuit = qml.compile(qaoa_probs_circuit, basis_set = allowed_gates)
        qaoa_probs_circuit = qml.compile(qaoa_probs_circuit, pipeline = [dispatched_transform])

        return qaoa_circuit, qaoa_probs_circuit
    

    def draw_qaoa_circuit(self):
        fig, ax = qml.draw_mpl(self.qaoa_circuit, expansion_strategy="device", decimals=2)(self.init_params)
        fig.savefig("qaoa_circuit.png")


    def get_latex_qaoa_circuit(self):
        dev = qml.device("qiskit.aer", wires=self.n_qubits)
        
        cost_hamiltonian = self.get_cost_hamiltonian()

        # Iterate over cost hamiltonian anc pick some elements
        coeffs, obs = cost_hamiltonian.coeffs, cost_hamiltonian.ops
        print("Coeffs: ", len(coeffs))
        print("Obs: ", obs)
        new_coeffs = [[], [], [], []]
        new_obs = [[], [], [], []]
        added_lengths = []

        #while len(new_coeffs) < 8:
        for coeff, op in zip(coeffs, obs):
            print("Length of op: ", len(op.wires))
            if len(op.wires) == 1 and len([v for v in added_lengths if v == 1]) < 2:
                new_coeffs[0].append(coeff)
                new_obs[0].append(op)
                added_lengths.append(1)
            elif len(op.wires) == 2 and len([v for v in added_lengths if v == 2]) < 2:
                new_coeffs[1].append(coeff)
                new_obs[1].append(op)
                added_lengths.append(2)
            elif len(op.wires) == 3 and len([v for v in added_lengths if v == 3]) < 2:
                new_coeffs[2].append(coeff)
                new_obs[2].append(op)
                added_lengths.append(3)
            elif len(op.wires) == 4 and len([v for v in added_lengths if v == 4]) < 2:
                new_coeffs[3].append(coeff)
                new_obs[3].append(op)
                added_lengths.append(4)

        new_coeffs = [item for sublist in new_coeffs for item in sublist]
        new_obs = [item for sublist in new_obs for item in sublist]

        cost_hamiltonian = qml.ops.op_math.LinearCombination(new_coeffs, new_obs)

        def qaoa_layer(gamma, alpha):
            qml.qaoa.cost_layer(gamma, cost_hamiltonian)
        
        @qml.qnode(dev)
        def qaoa_circuit(params):
            for wire in range(self.n_qubits):
                qml.Hadamard(wires=wire)
            qml.layer(qaoa_layer, 1, params[0], params[1])
            return qml.expval(cost_hamiltonian)
        
        qaoa_circuit(np.pi*np.random.rand(2, 1))

        latex = dev._circuit.draw(output="latex_source")

        return latex

    
    def solve_with_continuous_variables(self):
        if self.coskewness_tensor is None and self.cokurtosis_tensor is None:
            ef = EfficientFrontier(self.expected_returns, self.covariance_matrix)
            weights = ef.max_quadratic_utility(risk_aversion=self.risk_aversion)

            #print("Optimized Weights (considering variance and returns):")
            for asset, weight in weights.items():
                print(f"{self.stocks[asset]}: {weight:.2%}")

            #allocation, left_overs = ef.get_discrete_allocation(self.prices_now, self.budget)
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
            cost_matrix = self.get_cost_hamiltonian().matrix(wire_order=range(self.n_qubits))
            self.smallest_eigenvalues, self.smallest_eigenvectors, first_excited_energy, first_excited_state, eigenvalues = smallest_eigenpairs(cost_matrix)
        else:
            cost_matrix = self.get_cost_hamiltonian().sparse_matrix(wire_order=range(self.n_qubits))
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
        opt = qml.AdagradOptimizer(stepsize=0.01)
        #opt = qml.QNGOptimizer()
        params = self.init_params.copy()
        probs = self.qaoa_probs_circuit(params)
        total_steps = 500
        #attempts = 0
        #while True:
        #steps = 10
        for _ in range(total_steps):
            params = opt.step(self.qaoa_circuit, params)
        #total_steps += steps
        probs = self.qaoa_probs_circuit(params)
        #smallest_eigenvalue_now = self.qaoa_circuit(params)
        #most_probable_state = np.argsort(probs)[-1]
        #most_probable_state = int_to_bitstring(most_probable_state, self.n_qubits)
            #print(f"Most probable state: {most_probable_state} and smallest eigenvalue: {smallest_eigenvalue_now} and {self.smallest_bitstrings}")
        """
        if most_probable_state in self.smallest_bitstrings:
            break
        if total_steps > 1000:
            print("Optimization did not converge")
            print("Trying with a new initialization")
            self.init_params = np.pi*np.random.rand(2, self.layers, requires_grad=True)
            params = self.init_params.copy()
            total_steps = 0
            attempts += 1
        if attempts > 10:
            print("Optimization did not converge to the known optimal solution after ", attempts, " attempts.")
            print("Returning the best solution found so far")
            break
        """

        #probs = self.qaoa_probs_circuit(params)
        final_expectation_value = self.qaoa_circuit(params)
        two_most_probable_states = np.argsort(probs)[-2:]
        states_probs = [probs[i] for i in two_most_probable_states]
        two_most_probable_states = [int_to_bitstring(i, self.n_qubits) for i in two_most_probable_states]
        optimized_portfolios = bitstrings_to_optimized_portfolios(two_most_probable_states, self.assets_to_qubits)
        
        for portfolio in optimized_portfolios:
            self.satisfy_budget_constraint(portfolio)
        
        return two_most_probable_states, final_expectation_value, params, total_steps, states_probs, optimized_portfolios
    
    
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
        
        dev = qml.device('lightning.qubit', wires=self.n_qubits)
        
        cost_hamiltonian = self.get_cost_hamiltonian()
        #complete_graph = nx.complete_graph(self.n_qubits)
        mixer_hamiltonian = qml.qaoa.x_mixer(range(self.n_qubits)) #qml.qaoa.xy_mixer(complete_graph)

        def qaoa_layer(gamma, alpha):
            qml.qaoa.cost_layer(gamma, cost_hamiltonian)
            qml.qaoa.mixer_layer(alpha, mixer_hamiltonian)
        
        @qml.qnode(dev)
        def qaoa_circuit(params):
            gammas = params[:self.layers]
            alphas = params[self.layers:]
            for wire in range(self.n_qubits):
                qml.Hadamard(wires=wire)
            qml.layer(qaoa_layer, self.layers, gammas, alphas)
            return qml.expval(cost_hamiltonian)
        
        def objective_function(params):
            return float(qaoa_circuit(params))
        
        @qml.qnode(dev)
        def qaoa_probs_circuit(params):
            gammas = params[:self.layers]
            alphas = params[self.layers:]
            for wire in range(self.n_qubits):
                qml.Hadamard(wires=wire)
            qml.layer(qaoa_layer, self.layers, gammas, alphas)
            return qml.probs()

        initial_params = np.pi*np.random.rand(2, self.layers)
        # Make initial params 1-D array
        initial_params = np.concatenate((initial_params[0], initial_params[1]))
        print("Initial params: ", initial_params)
        es = cma.CMAEvolutionStrategy(initial_params, sigma0=0.1, options={"maxiter": maxiter})
        result = es.optimize(objective_function)
        optimized_params = result.result.xbest
        final_expectation_value = qaoa_circuit(optimized_params)
        probs = qaoa_probs_circuit(optimized_params)#[::-1]
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
        
        # Setup quantum device
        dev = qml.device('default.qubit', wires=self.n_qubits)
        
        # Get Hamiltonians
        cost_hamiltonian = self.get_cost_hamiltonian()
        mixer_hamiltonian = qml.qaoa.x_mixer(range(self.n_qubits))

        # Define QAOA layer
        def qaoa_layer(gamma, alpha):
            qml.qaoa.cost_layer(gamma, cost_hamiltonian)
            qml.qaoa.mixer_layer(alpha, mixer_hamiltonian)
        
        # Define circuit for expectation value
        @qml.qnode(dev)
        def qaoa_circuit(params):
            gammas = params[:self.layers]
            alphas = params[self.layers:]
            for wire in range(self.n_qubits):
                qml.Hadamard(wires=wire)
            qml.layer(qaoa_layer, self.layers, gammas, alphas)
            return qml.expval(cost_hamiltonian)
        
        # Define objective function for optimization
        # Track function evaluations
        func_evals = [0]  # Using a list for mutable counter
        
        def objective_function(params):
            func_evals[0] += 1
            return float(qaoa_circuit(params))
        
        # Define circuit for probability distribution
        @qml.qnode(dev)
        def qaoa_probs_circuit(params):
            gammas = params[:self.layers]
            alphas = params[self.layers:]
            for wire in range(self.n_qubits):
                qml.Hadamard(wires=wire)
            qml.layer(qaoa_layer, self.layers, gammas, alphas)
            return qml.probs()

        # Generate initial parameters
        initial_params = np.pi * np.random.rand(2, self.layers)
        initial_params = np.concatenate((initial_params[0], initial_params[1]))
        #print("Initial params: ", initial_params)
        
        # Define optimization options
        options = {'maxiter': maxiter}
        
        # Some optimizers require bounds or additional parameters
        bounds = None
        if optimizer in ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr']:
            bounds = [(0, 2*np.pi) for _ in range(len(initial_params))]
        
        # Extra options for specific optimizers
        if optimizer in ['trust-constr', 'trust-ncg', 'trust-krylov', 'trust-exact']:
            options['gtol'] = 1e-5
        
        # Some optimizers may require a Jacobian/Hessian, 
        # these are more demanding and should benchmarked the last
        jac = None
        hess = None
        if optimizer in ['Newton-CG', 'trust-ncg', 'trust-krylov', 'trust-exact', 'dogleg']:
            # For demonstration, we're using a finite difference approximation
            # In a real scenario, you might want to compute analytical derivatives
            jac = lambda x : approx_fprime(x, objective_function, 0.01)
            
            if optimizer in ['trust-exact', 'dogleg', 'trust-ncg', 'trust-krylov']:
                
                def hessian(x):
                    # Step size for finite difference approximation
                    epsilon = np.sqrt(np.finfo(float).eps)  # ~1.5e-8
                    n = len(x)
                    H = np.zeros((n, n))
                    
                    for i in range(n):
                        def partial_derivative_i(x):
                            return approx_fprime(x, objective_function, epsilon)[i]
                        
                        H[i] = approx_fprime(x, partial_derivative_i, epsilon)
                    
                    # Ensure symmetry (important for optimization stability)
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
        final_expectation_value = qaoa_circuit(optimized_params)
        probs = qaoa_probs_circuit(optimized_params)
        
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
            # Some optimizers don't provide nit attribute
            iterations = func_evals[0]  # Use our custom counter
        
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
            iterations,  # Use our tracked iterations
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
            return normalize_linear_combination(self.hamiltonian)
        return self.hamiltonian
    

    def solve_with_iterative_QAOA(self, max_layers = 10):
        # Start with layers = 1 and increase the number of layers until convergence and 
        # use the previous layers optimized params as initialization
        for layers in range(1, max_layers + 1):
            print(f"Trying with {layers} layers")
            self.layers = layers
            self.qaoa_circuit, self.qaoa_probs_circuit = self.get_QAOA_circuits()
            opt = qml.AdagradOptimizer(stepsize=0.7)
            if layers == 1:
                params = 0.01*np.random.rand(2, self.layers, requires_grad=True)
                #print("Initial params: ", params)
            else:
                params = np.array([[p for p in params[0]] + [params[0][-1]], [p for p in params[1]] + [params[1][-1]]], requires_grad=True)
                #print("New params: ", params)
            
            total_steps = 2000
            for _ in range(total_steps):
                params = opt.step(self.qaoa_circuit, params)
            
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

        return two_most_probable_states, final_expectation_value, params, total_steps, states_probs, optimized_portfolios
    
    
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
        opt = qml.AdamOptimizer(stepsize=0.01)
        params = self.init_params.copy()
        total_steps = 200
        epsilon = 1e-8
        target_probs = [epsilon]*2**self.total_qubits
        target_probs[int(target_bitstring, 2)] = 1 - (2**(self.total_qubits) - 1)*epsilon
        target_probs = np.array(target_probs)
        _, qaoa_probs_circuit = self.get_QAOA_circuits()
        
        def objective(params):
            # Maximize the overlap of the target bitstring with the most probable state
            probs = qaoa_probs_circuit(params)
            #print(type(probs[0]), type(target_probs[0]))
            return np.sum(target_probs*(np.log2(target_probs) - np.log2(probs))) #rel_entr(target_probs, probs)
            
        for i in range(total_steps):
            params, cost = opt.step_and_cost(objective, params)
            if i % 10 == 0:
                print(f"Cost: {cost}")
        
        return params
    
    def solve_exactly_with_lobpcg(self):
        from scipy.sparse.linalg import lobpcg

        k = 2
        n = 2**self.n_qubits
        rng = np.random.default_rng()
        X = rng.normal(size=(n, k))
        X = X.astype(np.float32)

        cost_matrix = self.get_cost_hamiltonian().sparse_matrix(wire_order=range(self.n_qubits))

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
