
import itertools
import math
import pickle
import numpy as np
import scipy

import pennylane as qml
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.typing import PostprocessingFn


def dicke_state_vector(n_qubits, k):
    """
    Build the explicit |D_n^k> state vector as a length-2^n numpy array.
    Wire 0 is the most-significant bit (matches qml.StatePrep convention).
    """
    if k < 0 or k > n_qubits:
        raise ValueError(f"Invalid Dicke params: n={n_qubits}, k={k}")
    state = np.zeros(2 ** n_qubits, dtype=float)
    norm = 1.0 / math.sqrt(math.comb(n_qubits, k))
    for indices in itertools.combinations(range(n_qubits), k):
        idx = 0
        for q in indices:
            idx |= (1 << (n_qubits - 1 - q))
        state[idx] = norm
    return state


def hamming_weight_indices(n_qubits, k):
    """Return all basis-state indices whose bitstring has Hamming weight k."""
    return [i for i in range(2 ** n_qubits) if bin(i).count("1") == k]


def replace_h_rz_h_with_rx(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    new_operations = []
    i = 0
    while i < len(tape.operations):
        op = tape.operations[i]

        # Detect pattern: H . RZ . H
        if (
            i + 2 < len(tape.operations)
            and op.name == "Hadamard"
            and tape.operations[i + 1].name == "RZ"
            and tape.operations[i + 2].name == "Hadamard"
            and op.wires == tape.operations[i + 1].wires == tape.operations[i + 2].wires
        ):
            rz_angle = tape.operations[i + 1].parameters[0]
            rx_angle = rz_angle  # RX(angle) = H . RZ(angle) . H
            new_operations.append(qml.RX(rx_angle, wires=op.wires[0]))

            # Skip the next two gates since they are replaced
            i += 3
        else:
            new_operations.append(op)
            i += 1

    # Create new transformed tape
    new_tape = tape.copy(operations=new_operations)

    def null_postprocessing(results):
        return results[0]

    return [new_tape], null_postprocessing


def decompose_ry(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Replace each RY(theta) with RZ(-pi/2), RX(theta), RZ(pi/2) (circuit order).

    Exact identity (no global phase): RY(t) = RZ(pi/2) @ RX(t) @ RZ(-pi/2),
    matching PennyLane's own _ry_to_rz_rx graph-decomposition rule.
    """
    new_operations = []
    for op in tape.operations:
        if op.name == "RY":
            w = op.wires[0]
            new_operations.append(qml.RZ(-np.pi / 2, wires=w))
            new_operations.append(qml.RX(op.parameters[0], wires=w))
            new_operations.append(qml.RZ(np.pi / 2, wires=w))
        else:
            new_operations.append(op)

    new_tape = tape.copy(operations=new_operations)

    def null_postprocessing(results):
        return results[0]

    return [new_tape], null_postprocessing


def smallest_eigenpairs(A, filename = None):
    """
    Return the smallest eigenvalues and eigenvectors of a matrix A
    Returns always at least two eigenvalues and eigenvectors, 
    even if the second solution is not optimal.
    The non-zero difference between the two smallest eigenvalues 
    can describe hardness of the optimization problem.
    """

    eigenvalues, eigenvectors = scipy.linalg.eig(A)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    idx = np.argsort(eigenvalues)
    smallest_eigenvalues = []
    smallest_eigenvectors = []

    smallest_eigenvalue = eigenvalues[idx[0]]
    smallest_eigenvalues.append(smallest_eigenvalue)
    smallest_eigenvectors.append(eigenvectors[:, idx[0]])

    first_excited_energy = None
    first_excited_state = None
    
    # Find all smallest eigenvalues and eigenvectors
    for i in range(1, len(eigenvalues)):
        if eigenvalues[idx[i]] == smallest_eigenvalue:
            smallest_eigenvalues.append(eigenvalues[idx[i]])
            smallest_eigenvectors.append(eigenvectors[:, idx[i]])
        else:
            first_excited_energy = eigenvalues[idx[i]]
            first_excited_state = eigenvectors[:, idx[i]]
            break
    
    if filename is not None:
        with open(filename, "wb") as f:
            pickle.dump([eigenvalues, eigenvectors], f)
    
    return smallest_eigenvalues, smallest_eigenvectors, first_excited_energy, first_excited_state, eigenvalues

def smallest_sparse_eigenpairs(A):
    """Smallest eigenpair of the (diagonal) HUBO cost Hamiltonian.

    The cost Hamiltonian is built purely from Identity/PauliZ products, so its
    matrix is diagonal in the computational basis: the eigenvalues ARE the
    diagonal entries and the ground state is the basis vector at the minimum.
    Reading the diagonal is exact and instant, avoiding the pathologically slow
    `eigsh(..., which='SA')` ARPACK path on 2**n matrices (which effectively
    hangs for n >= 14).

    Returns (smallest_eigenvalues, smallest_eigenvectors, eigenvalues) with the
    same contract as before: `eigenvalues` is the full diagonal (the spectrum).
    A single ground eigenpair is returned (strict parity with the old path, whose
    float-tie degeneracy check at `eigenvalues[0] == eigenvalues[1]` never fired).

    Raises ValueError if A is not diagonal/real -- that signals a Hamiltonian
    construction bug, and `solve_exactly` already routes a raise to its
    `solve_exactly_with_lobpcg` fallback, which converges fine on a diagonal
    matrix (whereas silently re-running eigsh would just hang again).
    """
    diag = np.asarray(A.diagonal()).ravel()  # handles the one implicit-zero diagonal entry

    # Diagonality + real guard: value-based and O(nnz), no tolerance to tune. An
    # explicit *stored* zero off the diagonal (possible from coefficient
    # cancellation) does not trip it -- only a nonzero off-diagonal value does.
    coo = A.tocoo()
    off = coo.row != coo.col
    if off.any() and np.abs(coo.data[off]).max() > 0:
        raise ValueError("cost Hamiltonian matrix is not diagonal; "
                         "refusing to treat its diagonal as the spectrum")
    if np.abs(diag.imag).max() > 1e-9:
        raise ValueError("cost Hamiltonian diagonal has a non-negligible imaginary "
                         "part; expected a real (Z-only) Hamiltonian")
    d = np.real(diag)

    # Ground state: single argmin -> one-hot basis vector at the minimum.
    idx = int(np.argmin(d))
    onehot = np.zeros_like(d)
    onehot[idx] = 1.0
    smallest_eigenvalues = [d[idx]]
    smallest_eigenvectors = [np.array([int(x) for x in onehot])]

    return smallest_eigenvalues, smallest_eigenvectors, d


def bitstring_to_int(bit_string_sample):
    if type(bit_string_sample[0]) == str:
        bit_string_sample = np.array([int(i) for i in bit_string_sample])
    return int(2 ** np.arange(len(bit_string_sample)) @ bit_string_sample)


def int_to_bitstring(int_sample, n_qubits):
    bits = np.array([int(i) for i in format(int_sample, f'0{n_qubits}b')])
    return "".join([str(i) for i in bits])


def basis_vector_to_bitstring(basis_vector):
    assert np.sum(basis_vector) == 1, "Input must be a basis vector"
    index = np.argmax(basis_vector)
    num_qubits = max(int(np.log2(len(basis_vector))), 1)
    bitstring = format(index, f'0{num_qubits}b')
    #bitstring = np.array(list(np.binary_repr(index).zfill(num_qubits)))
    bitstring = [int(i) for i in bitstring]
    return bitstring

def bitstrings_to_optimized_portfolios(bitstrings, assets_to_qubits):
    """
    Given a bitstring, return the portfolio that corresponds to the bitstring with log encoding
    """
    portfolios = []
    for bitstring in bitstrings:
        portfolio = {}
        for asset, qubits in assets_to_qubits.items():
            bits = [bitstring[q] for q in qubits]
            portfolio[asset] = bitstring_to_int(bits)
        for asset in assets_to_qubits.keys():
            if asset not in portfolio.keys():
                portfolio[asset] = 0
        portfolios.append(portfolio)
    return portfolios

def normalize_linear_combination(lin_comb):
    """Normalize a PennyLane LinearCombination operation."""
    coeffs, ops = lin_comb.terms() # Extract coefficients and operators
    norm_factor = sum(abs(c) for c in coeffs)  # Compute sum of absolute values

    if norm_factor == 0:
        raise ValueError("Cannot normalize: all coefficients are zero.")

    normalized_coeffs = [c / norm_factor for c in coeffs]
    return qml.ops.op_math.LinearCombination(normalized_coeffs, ops)