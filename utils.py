
import pickle
import numpy as np
import scipy

import pennylane as qml
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.typing import PostprocessingFn

def extract_from_latex(latex_source):
        """
        Extract characters from each line starting with '\nghost' up to the tenth '&' character.
        
        Args:
            latex_source (str): The LaTeX source code
            
        Returns:
            list: Lines extracted according to the specified rule
        """
        depth = 100
        extracted_lines = []
        
        # Split the latex source into lines
        lines = latex_source.split('\n')
        
        # Process each line
        for line in lines:
            if line.strip().startswith('\\nghost'):
                # Count the occurrences of '&'
                amp_positions = [pos for pos, char in enumerate(line) if char == '&']
                
                # Check if there are at least 10 '&' characters
                if len(amp_positions) >= depth:
                    # Extract up to the 10th '&'
                    extracted_portion = line[:amp_positions[depth - 1]]
                    extracted_lines.append(extracted_portion + '\\\ \n')
                else:
                    # If fewer than 10 '&' characters, take the whole line
                    extracted_lines.append(line)
            else:
                extracted_lines.append(line + '\n')
        
        return extracted_lines



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
    
    # Get the smallest eigenvalues and eigenvectors
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(A, k=3, which='SA')
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    smallest_eigenvalues = []
    smallest_eigenvectors = []
    
    smallest_eigenvalues.append(eigenvalues[0])
    smallest_eigenvectors.append(eigenvectors[:, 0])
    
    # Check if there are degenerate eigenvalues
    if eigenvalues[0] == eigenvalues[1]:
        smallest_eigenvalues.append(eigenvalues[1])
        smallest_eigenvectors.append(eigenvectors[:, 1])
    
    # For each eigenvector make the largest element to be one and others to be zero
    for i in range(len(smallest_eigenvectors)):
        smallest_eigenvectors_new = np.zeros_like(smallest_eigenvectors[i])
        index = np.argmax(smallest_eigenvectors[i])
        smallest_eigenvectors_new[index] = 1.0
        smallest_eigenvectors[i] = np.array([int(i) for i in smallest_eigenvectors_new])
    
    return smallest_eigenvalues, smallest_eigenvectors, eigenvalues


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