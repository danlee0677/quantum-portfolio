"""Smoke test for the PennyLane -> Qiskit migration."""
import numpy as np
from portfolio_utils import normalize_hamiltonian, bitstring_to_int, int_to_bitstring
from qiskit.quantum_info import SparsePauliOp

# Test 1: normalize_hamiltonian
h = SparsePauliOp.from_list([('ZZ', 2.0), ('IX', -1.0), ('XI', 3.0)])
h_norm = normalize_hamiltonian(h)
assert np.isclose(np.sum(np.abs(h_norm.coeffs)), 1.0), "Normalization failed"
print("PASS: normalize_hamiltonian")

# Test 2: bitstring helpers
assert bitstring_to_int([1, 0, 1]) == 5, "bitstring_to_int failed"
assert int_to_bitstring(5, 4) == "1010", "int_to_bitstring failed"  # Qiskit little-endian: qubit 0 (LSB) first
print("PASS: bitstring helpers")

# Test 3: Hamiltonian construction via the class (small 2-asset problem)
from portfolio_hubo_qaoa_solver import HigherOrderPortfolioQAOA

stocks = ["A", "B"]
prices_now = {"A": 100, "B": 200}
expected_returns = [0.1, 0.2]
covariance_matrix = [[0.04, 0.01], [0.01, 0.09]]

portfolio = HigherOrderPortfolioQAOA(
    stocks=stocks,
    prices_now=prices_now,
    expected_returns=expected_returns,
    covariance_matrix=covariance_matrix,
    budget=500,
    max_qubits=15,
    log_encoding=True,
    risk_aversion=0.1,
    strict_budget_constraint=False,
    lambda_budget=1
)
print(f"PASS: HigherOrderPortfolioQAOA constructed ({portfolio.get_n_qubits()} qubits, {portfolio.get_layers()} layers)")

# Test 4: Hamiltonian is a valid SparsePauliOp
h = portfolio.get_cost_hamiltonian()
assert isinstance(h, SparsePauliOp), f"Expected SparsePauliOp, got {type(h)}"
print(f"PASS: Hamiltonian is SparsePauliOp with {len(h)} terms")

# Test 5: Hamiltonian matrix is Hermitian
mat = h.to_matrix()
assert np.allclose(mat, mat.conj().T), "Hamiltonian matrix is not Hermitian"
print("PASS: Hamiltonian matrix is Hermitian")

# Test 6: solve_exactly
result = portfolio.solve_exactly()
eigenvalues, bitstrings = result[0], result[1]
print(f"PASS: solve_exactly (ground state energy = {eigenvalues[0]:.6f}, bitstring = {bitstrings[0]})")

# Test 7: QAOA circuit evaluation
qaoa_circuit, qaoa_probs = portfolio.get_QAOA_circuits()
n_qubits = portfolio.get_n_qubits()
layers = portfolio.get_layers()
test_params = np.pi * np.random.rand(2 * layers)
expval = qaoa_circuit(test_params)
probs = qaoa_probs(test_params)
assert isinstance(expval, float), f"Expected float, got {type(expval)}"
assert len(probs) == 2**n_qubits, f"Expected {2**n_qubits} probs, got {len(probs)}"
assert np.isclose(np.sum(probs), 1.0), f"Probs don't sum to 1: {np.sum(probs)}"
print(f"PASS: QAOA circuit evaluation (expval={expval:.6f}, probs sum={np.sum(probs):.10f})")

print("\n=== ALL SMOKE TESTS PASSED ===")
