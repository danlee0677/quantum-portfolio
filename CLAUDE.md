# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Implementation for the paper: **"Higher-Order Portfolio Optimization with Quantum Approximate Optimization Algorithm"** (Uotila, Ripatti, Zhao — Aalto University & University of Helsinki, arXiv:2509.01496). The accompanying paper PDF is at `../졸업연구/HOPO_with_QAOA.pdf`.

This is the first quantum formulation of portfolio optimization that includes higher-order moments (skewness and kurtosis). It formulates the problem as a Higher-Order Unconstrained Binary Optimization (HUBO) problem and solves it via QAOA on simulated quantum circuits. The key finding is that HUBO solutions often produce better portfolio allocations than the classical continuous-variable baseline with integer programming discretization.

## Running Experiments

This project is managed by **uv** (`pyproject.toml`, `uv.lock`, `.python-version`). Run Python via `uv run`, which auto-syncs the environment first — do **not** use conda.

`experiments.py` runs one of two QAOA algorithms, selected with `--method` (the
shared classical + exact baselines are written either way). It takes an
**inclusive, 0-based experiment range** `start end` (no longer a batch split):
```bash
# Original paper method (integer-HUBO + raw QAOA) — default
uv run python experiments.py <start> <end> --method hopo
# Cardinality-selection method (ring XY-mixer QAOA)
uv run python experiments.py <start> <end> --method ring_xy
# e.g. experiments #1–4:
uv run python experiments.py 1 4 --method ring_xy
```

**Choosing the cardinality K (`--method ring_xy` only).** K is no longer brute-forced
by default. A cheap, purely-classical selector picks K so the expensive fixed-K
QAOA runs only on the top candidate(s):
1. **LASSO localizer** — an L1-penalized continuous higher-moment solve (short
   λ-path) gives a ballpark `K_hat` and an asset ranking (`RingXYCardinalityQAOA.lasso_localize_K`).
2. **Classical neighborhood scan** — K in `[K_hat-w, K_hat+w]` is scored by
   allocating the top-K assets with the existing weighted objective + discrete
   allocator (`select_K_classically`, reusing `_allocate_on_subset`). No QAOA,
   no subset enumeration, so it scales to large N.
3. **Fixed-K QAOA** — runs only on the top `--k-hedge` classical K; `best_K` is the
   QAOA-run K with the best post-allocation objective (same rule as before).

Flags: `--k-selector {lasso,sweep,both}` (default `lasso`; `sweep` = original
brute force over `[k_min, N]`; `both` = full sweep ground truth + records the
lasso pick for offline accuracy), `--k-neighborhood` (default 2),
`--k-hedge` (default 2), `--k-min` (default 2).
```bash
# default: cheap classical K-selector, QAOA on top-2 K
uv run python experiments.py 1 4 --method ring_xy
# reproduce the original brute-force K sweep
uv run python experiments.py 1 4 --method ring_xy --k-selector sweep
# validation: full sweep + record what lasso would have picked
uv run python experiments.py 1 4 --method ring_xy --k-selector both
```

> NOTE: the SLURM scripts (`run.sh`, `run_ring_xy.sh`) still pass the old
> `batch_num total_batches` arguments and need updating to the `start end`
> signature before `sbatch` will work.

All output goes to `results/`, one file per run with the experiment range and a
`YYYYMMDD_HHMMSS` timestamp in the name, so re-running a range never overwrites
an earlier run. Results are saved incrementally (after each experiment) and a
later run skips any experiment id already present in an earlier `results/` file
for the same method:
- `--method hopo` → `results/portfolio_optimization_exp_<optimizer>_<lambda>_<start>_<end>_<timestamp>.json`
  (contains `qaoa_solution`, no `cardinality_qaoa_solution`).
- `--method ring_xy` → `results/ring_xy_exp_<start>_<end>_<timestamp>.json`
  (contains `cardinality_qaoa_solution`, no `qaoa_solution`). Beyond `per_K` and
  `best_K`, `cardinality_qaoa_solution` now also records the K-selector trail:
  `k_selector`, `qaoa_K_run` (K actually given to QAOA), and — for the lasso/both
  selectors — `k_hat`, `classical_K_scan` (the cheap per-K classical curve),
  `lasso_support_path`, and (in `both` mode) `lasso_picked_K`. With the default
  `lasso` selector, `per_K` holds only the hedged K (≤ `--k-hedge` entries)
  instead of the full sweep.
Both files also carry the shared `continuous_variables_solution[_unconstrained]`
and `exact_solution` baselines. Per-experiment JSON keys stay 0-based.

`profile.py` accepts the same optional `start end` range (e.g. `profile.py specs
1 4`). For the cardinality method it resolves `best_K` by pulling each requested
experiment from the **newest** `results/ring_xy_exp_*.json` file that contains
it (per-experiment, newest-first merge); `--results-json PATH` forces a single
file, and `--rederive-k` recomputes `best_K` via a K sweep.

**HOPO-with-QAOA comparison from JSON (`convergence` subcommand).** By default
the integer-HUBO baseline is recomputed live every run (`--hopo-source live`,
the expensive `solve_with_qaoa_cma_es` + `solve_exactly`). `--hopo-source json`
instead loads those convergence metrics from saved
`results/portfolio_optimization_exp_*.json` (independent of the ring_xy best_K
stream, via `build_hopo_results` + `hopo_row_from_json`): a pure
`--method hopo --hopo-source json` run skips the yfinance download and the HOPO
build entirely. `final_expectation_value`/`post_objective`/`training_history`
come from `qaoa_solution`, `E_min`/`E_max` from `exact_solution.spectrum`;
`wall_clock_seconds` is `None` and each row is tagged `"source": "json"`.
`--hopo-results-json PATH` forces one file. When an experiment is missing from
the JSON (or its entry is unusable), that experiment falls back to a live
recompute. ring_xy `best_K` resolution is unchanged.
```bash
# load HOPO rows from saved JSON instead of recomputing (falls back to live)
uv run python profile.py convergence 0 4 --method hopo --hopo-source json
```

**Cardinality rows from JSON (`--card-source json`, `convergence` only).** The
mirror of `--hopo-source` for the ring_xy side: instead of re-running
`solve_with_qaoa_cardinality(best_K)` live (default `--card-source live`),
`card_row_from_json` loads the `best_K` entry from
`cardinality_qaoa_solution.per_K` in the same merged `results/ring_xy_exp_*.json`
stream that `best_K` resolution already uses (`--results-json PATH` forces one
file). `wall_clock_seconds` is `None` (the ring_xy JSON has no per-K timing) and
rows are tagged `"source": "json"` (live rows: `"source": "live"`). Missing or
unusable entries fall back to a live run per experiment; `--sweep`/`--rederive-k`
bypass the JSON path entirely. A pure
`--method both --hopo-source json --card-source json` run skips the yfinance
download and all solver builds.
```bash
# load both methods' convergence rows from saved JSON (no QAOA, no download)
uv run python profile.py convergence 0 4 --method both --hopo-source json --card-source json
```

**Migrating the author's results (`migrate_author_hopo_results.py`).** The
previous author's HOPO results in `results_old/cmaes_hubo_results/lambda_<L>/`
(one `portfolio_optimization_results_batch_*.json` per experiment; `lambda_1` =
canonical `lambda_budget=1.0`) are converted into one current-format file in
`results/` so `--hopo-source json` can ingest them. Migration merges the batch
files and adds the two fields the current pipeline writes
(`hyperparams.{optimizer,lambda_budget}`,
`continuous_variables_solution_unconstrained=null`); deterministic baselines are
left untouched. The `<stamp>` arg is a `YYYYMMDD_HHMMSS` token for newest-first
sorting.
```bash
uv run python migrate_author_hopo_results.py 20260610_120000  # -> results/portfolio_optimization_exp_CMAES_1.0_0_99_*.json
```

**Verifying equivalence (`verify_hopo_equivalence.py`).** A lightweight drift
check (run on 1–2 small ids on local hardware) that rebuilds each problem from
the author's stored hyperparams, runs the current pipeline at `lambda_budget=1.0`,
and diffs three tiers: **data parity** (fresh yfinance `prices_now` vs stored),
**deterministic** (exact eigen-spectrum + continuous baseline, must match), and a
**QAOA band** (CMA-ES is unseeded, so only checked for plausibility). A
deterministic mismatch is a code-drift FAIL *only when data matched*; because
yfinance re-adjusts historical prices over time, data typically diverges and the
verdict is **INCONCLUSIVE** — which is exactly why the migrated author JSON is
the only faithful baseline (the author's exact returns are not recoverable).
```bash
uv run python verify_hopo_equivalence.py 0 1 --qaoa-runs 2
```

## Dependencies

Managed via uv (`pyproject.toml` + `uv.lock`). Add packages with `uv add <pkg>`, sync with `uv sync`.

`numpy`, `scipy`, `pennylane`, `pypfopt` (PyPortfolioOpt), `yfinance`, `cma` (CMA-ES optimizer), `cvxpy` (for discretization integer program)

## Architecture

### Core pipeline (how a single experiment runs)

1. **Data generation** — `generate_portfolio_experiments.ipynb` creates random portfolio problems from real stock data (yfinance) and writes them to `experiments_data.json`.

2. **Experiment runner** — `experiments.py` loads experiment definitions, selects the requested inclusive `start end` id range, and for each experiment:
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
   - **Cardinality-selection QAOA** (the benchmarked contribution) — `RingXYCardinalityQAOA.solve_with_qaoa_cardinality()`: a y_i ∈ {0,1} HUBO that picks K of N stocks (Dicke initial state, ring XY-mixer, no budget penalty), then a classical integer-program allocator on the chosen K-subset. The cardinality K is chosen by a cheap classical selector (LASSO localizer + neighborhood scan, `select_K_classically`) so the fixed-K QAOA runs only on the top candidate(s); see "Choosing the cardinality K" above. The original brute-force K sweep is still available via `--k-selector sweep`.

### Key modules

- **`portfolio_hubo_qaoa_light.py`** — Class `HigherOrderPortfolioQAOA`, the **original paper method only** (integer-HUBO + raw QAOA). Handles HUBO construction from portfolio moments, integer-to-binary variable encoding (log encoding), budget penalty in the cost Hamiltonian, conversion to Ising Hamiltonian, QAOA circuit construction (X-mixer, Hadamard init), and optimization. Entry points: `solve_with_qaoa_cma_es`, `solve_with_qaoa_scipy`, `solve_with_qaoa`, `solve_with_iterative_QAOA`, `solve_exactly`, `solve_exactly_with_lobpcg`, `solve_with_continuous_variables`, `solve_with_continuous_variables_unconstrained`; circuit helper `build_integer_hubo_circuit`. The classical/exact baselines also live here.

- **`ring_xy.py`** — Stand-alone class `RingXYCardinalityQAOA`, the **cardinality-selection method only** (the benchmarked contribution). A y_i ∈ {0,1} HUBO that picks K of N stocks: Dicke initial state, ring XY-mixer (`qml.qaoa.xy_mixer` on a cycle graph), no budget penalty, then a classical IP allocator on the K-subset. Methods: `construct_selection_hubo_bin`, `_slice_problem`, `_allocate_on_subset`, `build_cardinality_circuit`, `solve_with_qaoa_cardinality`, plus the classical K-selector `lasso_localize_K` / `select_K_classically` (LASSO localizer + neighborhood scan that picks K without per-K QAOA). Shares no algorithm code with `HigherOrderPortfolioQAOA` (it keeps its own copies of `get_objective_value` / `cma_result_to_dict`); reuses shared helpers from `utils.py` and the classical baselines (`HigherMomentPortfolioOptimizer`, `EfficientFrontier`, `DiscreteAllocation`).

- **`portfolio_higher_moments_classical.py`** — `HigherMomentPortfolioOptimizer` class providing classical baselines using scipy optimization with higher-moment objectives (mean-variance-skewness-kurtosis). Variants: constrained (`optimize_portfolio_with_higher_moments`), penalty-unconstrained (`..._unconstrained`), and L1-penalized (`..._l1`, the sparsity localizer used by the ring_xy K-selector).

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
