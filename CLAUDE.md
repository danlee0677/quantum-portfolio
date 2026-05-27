# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Implementation for the paper: **"Higher-Order Portfolio Optimization with Quantum Approximate Optimization Algorithm"** (Uotila, Ripatti, Zhao — Aalto University & University of Helsinki, arXiv:2509.01496). The accompanying paper PDF is at `../졸업연구/HOPO_with_QAOA.pdf`.

This is the first quantum formulation of portfolio optimization that includes higher-order moments (skewness and kurtosis). It formulates the problem as a Higher-Order Unconstrained Binary Optimization (HUBO) problem and solves it via QAOA on simulated quantum circuits. The key finding is that HUBO solutions often produce better portfolio allocations than the classical continuous-variable baseline with integer programming discretization.

## Running Experiments

Experiments are designed to run on a SLURM cluster via `run.sh`:
```bash
# Local single-batch run
python experiments.py <batch_num> <total_batches>

# SLURM submission (100 parallel jobs)
sbatch run.sh
```

Results are saved incrementally to JSON files (`portfolio_optimization_batch_*.json`) so progress is not lost if a job is interrupted.

## Dependencies

`numpy`, `scipy`, `pennylane`, `pypfopt` (PyPortfolioOpt), `yfinance`, `cma` (CMA-ES optimizer), `cvxpy` (for discretization integer program)

## Architecture

### Core pipeline (how a single experiment runs)

1. **Data generation** — `generate_portfolio_experiments.ipynb` creates random portfolio problems from real stock data (yfinance) and writes them to `experiments_data.json`.

2. **Experiment runner** — `experiments.py` loads experiment definitions, splits them into batches (for SLURM parallelism), and for each experiment:
   - Downloads stock data, computes expected returns (`pypfopt`) and higher-order moments
   - Constructs and solves the problem via three methods, comparing results

3. **Four solution methods** (all invoked through `HigherOrderPortfolioQAOA` in `portfolio_hubo_qaoa_light.py`):
   - **Constrained classical** — `solve_with_continuous_variables()` optimizes continuous weights with budget constraint (1^T w = 1), then discretizes via integer programming (Eq. 8 in paper)
   - **Unconstrained classical** — `solve_with_continuous_variables_unconstrained()` optimizes with penalty term (1^T w - 1)^2 instead of hard constraint, closer to the HUBO formulation (Eq. 19 in paper)
   - **Exact HUBO** — `solve_exactly()` constructs the full Hamiltonian matrix and finds the ground state via eigendecomposition (sparse solver `solve_exactly_with_lobpcg()` for 14-15 qubits)
   - **QAOA** — `solve_with_qaoa_cma_es()` or `solve_with_qaoa_scipy()` runs parameterized quantum circuits (PennyLane) optimized with CMA-ES or scipy optimizers

### Key modules

- **`portfolio_hubo_qaoa_light.py`** — Central class `HigherOrderPortfolioQAOA`. Handles the full pipeline: HUBO construction from portfolio moments, integer-to-binary variable encoding (log encoding), conversion to Ising Hamiltonian, QAOA circuit construction (PennyLane), and optimization. This is the largest and most complex file.

- **`portfolio_higher_moments_classical.py`** — `HigherMomentPortfolioOptimizer` class providing classical baselines using scipy optimization with higher-moment objectives (mean-variance-skewness-kurtosis).

- **`coskewness_cokurtosis.py`** — Computes normalized coskewness (3rd order) and excess cokurtosis (4th order) tensors from return data.

- **`utils.py`** — Bitstring/integer conversion helpers, eigenpair computation, PennyLane tape transform (`replace_h_rz_h_with_rx` simplifies H-RZ-H sequences to RX gates).

### HUBO encoding scheme

The `HigherOrderPortfolioQAOA` constructor performs these transformations in sequence:
1. `construct_cost_hubo_int()` — builds cost function with integer variables from portfolio moments
2. Budget constraint added (quadratic penalty or strict with slack variables)
3. `replace_integer_variables_with_binary_variables()` — log-encodes integers as binary (each integer x = Σ 2^k · b_k)
4. `simplify_cost_hubo_bin()` — applies b² = b simplification
5. `cost_hubo_bin_to_ising_hamiltonian()` — converts to Pauli-Z Hamiltonian for QAOA

### Mathematical formulation (paper references)

The cost function (Eq. 15): `min q2·K(z) - q1·S(z) + q0·z^T·c·z - μ^T·z + λ·(z^T·p^τ - C)^2`
- q0 = risk_aversion/2, q1 = risk_aversion/6, q2 = risk_aversion/24 (Edgeworth expansion, Sec. II-F)
- Integer-to-binary log encoding (Eq. 17): `z = (N+1-2^M)·y_M + Σ 2^n·y_n`
- Binary-to-spin mapping: `x_i = (1 - s_i)/2` to get Pauli-Z Hamiltonian (Sec. III-C)
- HUBO Hamiltonian has terms up to degree 4 (from cokurtosis tensor)

### Experimental dataset

- 100 random portfolio problems from Dow Jones Industrial Average (30 companies)
- 2-10 randomly sampled companies per problem, random budgets (capped at 6000)
- 10 problems for each qubit count from 6-15 qubits
- Stock data: Jan 2015 to Jan 2025 (10 years) from yfinance
- Evaluation metric: min-max normalized objective f(z) vs. budget utilization (Fig. 6 in paper)

### Configuration

Key parameters in `experiments.py`:
- `classical_optimizer` — optimizer for QAOA parameters (`"CMAES"` default; also tested Powell, SLSQP, COBYLA, Nelder-Mead, L-BFGS-B — CMA-ES performed best, see Fig. 4 in paper)
- `lambda_budget` — weight of the budget constraint penalty (explored: 0.001 to 1000; values >1 generally best)
- `risk_aversion` — set to 0.1 in experiments (controls moment weights via Edgeworth expansion)
- `max_qubits` — upper limit on total qubits (15 in the paper experiments)
- QAOA layers are set to `min(10, n_qubits)` by default
- CMA-ES uses σ=0.1 with default hyperparameters otherwise
