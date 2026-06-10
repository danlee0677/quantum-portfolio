# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Implementation for the paper: **"Higher-Order Portfolio Optimization with Quantum Approximate Optimization Algorithm"** (Uotila, Ripatti, Zhao — Aalto University & University of Helsinki, arXiv:2509.01496). The accompanying paper PDF is at `../졸업연구/HOPO_with_QAOA.pdf`.

This is the first quantum formulation of portfolio optimization that includes higher-order moments (skewness and kurtosis). It formulates the problem as a Higher-Order Unconstrained Binary Optimization (HUBO) problem and solves it via QAOA on simulated quantum circuits. The key finding is that HUBO solutions often produce better portfolio allocations than the classical continuous-variable baseline with integer programming discretization.

## Running Experiments

This project is managed by **uv** (`pyproject.toml`, `uv.lock`, `.python-version`). Run Python via `uv run`, which auto-syncs the environment first — do **not** use conda.

Experiments are designed to run on a SLURM cluster. `experiments.py` runs one of
two QAOA algorithms, selected with `--method` (the shared classical + exact
baselines are written either way):
```bash
# Original paper method (integer-HUBO + raw QAOA) — default
uv run python experiments.py <batch_num> <total_batches> --method hopo
# Cardinality-selection method (ring XY-mixer QAOA)
uv run python experiments.py <batch_num> <total_batches> --method ring_xy

# SLURM submission (100 parallel jobs)
sbatch run.sh          # --method hopo
sbatch run_ring_xy.sh  # --method ring_xy
```

Results are saved incrementally so progress is not lost if a job is interrupted:
- `--method hopo` → `portfolio_optimization_batch_<optimizer>_<lambda>_<batch>.json`
  (contains `qaoa_solution`, no `cardinality_qaoa_solution`).
- `--method ring_xy` → `ring_xy_batch_<batch>.json`
  (contains `cardinality_qaoa_solution`, no `qaoa_solution`).
Both files also carry the shared `continuous_variables_solution[_unconstrained]`
and `exact_solution` baselines.

## Dependencies

Managed via uv (`pyproject.toml` + `uv.lock`). Add packages with `uv add <pkg>`, sync with `uv sync`.

`numpy`, `scipy`, `pennylane`, `pypfopt` (PyPortfolioOpt), `yfinance`, `cma` (CMA-ES optimizer), `cvxpy` (for discretization integer program)

## Architecture

### Core pipeline (how a single experiment runs)

1. **Data generation** — `generate_portfolio_experiments.ipynb` creates random portfolio problems from real stock data (yfinance) and writes them to `experiments_data.json`.

2. **Experiment runner** — `experiments.py` loads experiment definitions, splits them into batches (for SLURM parallelism), and for each experiment:
   - Downloads stock data, computes expected returns (`pypfopt`) and higher-order moments
   - Constructs and solves the problem via three methods, comparing results

3. **Solution methods.** The classical + exact baselines and the original-paper
   QAOA are invoked through `HigherOrderPortfolioQAOA`
   (`portfolio_hubo_qaoa_light.py`); the cardinality-selection QAOA lives in the
   separate `RingXYCardinalityQAOA` class (`ring_xy.py`):
   - **Constrained classical** — `solve_with_continuous_variables()` optimizes continuous weights with budget constraint (1^T w = 1), then discretizes via integer programming (Eq. 8 in paper)
   - **Unconstrained classical** — `solve_with_continuous_variables_unconstrained()` optimizes with penalty term (1^T w - 1)^2 instead of hard constraint, closer to the HUBO formulation (Eq. 19 in paper)
   - **Exact HUBO** — `solve_exactly()` constructs the full Hamiltonian matrix and finds the ground state via eigendecomposition (sparse solver `solve_exactly_with_lobpcg()` for 14-15 qubits)
   - **HOPO + raw QAOA** (paper baseline) — `solve_with_qaoa_cma_es()` or `solve_with_qaoa_scipy()` runs parameterized quantum circuits (PennyLane) optimized with CMA-ES or scipy optimizers
   - **Cardinality-selection QAOA** (the benchmarked contribution) — `RingXYCardinalityQAOA.solve_with_qaoa_cardinality()`: a y_i ∈ {0,1} HUBO that picks K of N stocks (Dicke initial state, ring XY-mixer, no budget penalty), then a classical integer-program allocator on the chosen K-subset

### Key modules

- **`portfolio_hubo_qaoa_light.py`** — Class `HigherOrderPortfolioQAOA`, the **original paper method only** (integer-HUBO + raw QAOA). Handles HUBO construction from portfolio moments, integer-to-binary variable encoding (log encoding), budget penalty in the cost Hamiltonian, conversion to Ising Hamiltonian, QAOA circuit construction (X-mixer, Hadamard init), and optimization. Entry points: `solve_with_qaoa_cma_es`, `solve_with_qaoa_scipy`, `solve_with_qaoa`, `solve_with_iterative_QAOA`, `solve_exactly`, `solve_exactly_with_lobpcg`, `solve_with_continuous_variables`, `solve_with_continuous_variables_unconstrained`; circuit helper `build_integer_hubo_circuit`. The classical/exact baselines also live here.

- **`ring_xy.py`** — Stand-alone class `RingXYCardinalityQAOA`, the **cardinality-selection method only** (the benchmarked contribution). A y_i ∈ {0,1} HUBO that picks K of N stocks: Dicke initial state, ring XY-mixer (`qml.qaoa.xy_mixer` on a cycle graph), no budget penalty, then a classical IP allocator on the K-subset. Methods: `construct_selection_hubo_bin`, `_slice_problem`, `_allocate_on_subset`, `build_cardinality_circuit`, `solve_with_qaoa_cardinality`. Shares no algorithm code with `HigherOrderPortfolioQAOA` (it keeps its own copies of `get_objective_value` / `cma_result_to_dict`); reuses shared helpers from `utils.py` and the classical baselines (`HigherMomentPortfolioOptimizer`, `EfficientFrontier`, `DiscreteAllocation`).

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
