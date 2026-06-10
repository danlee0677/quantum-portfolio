## Main files in this repo

1. The notebook `generate_portfolio_experiments.ipynb` generates random portfolio optimization problems based on the real stock data from yfinance, written to `experiments_data.json`.
2. `experiments.py` — the main experiment runner. Selects one of two QAOA algorithms via `--method`.
3. `portfolio_hubo_qaoa_light.py` — class `HigherOrderPortfolioQAOA`, the **original paper method** (integer-HUBO + raw QAOA, X-mixer, budget penalty). Also holds the shared classical/exact baselines.
4. `ring_xy.py` — class `RingXYCardinalityQAOA`, the **new cardinality-selection method** (picks K of N stocks with a Dicke initial state + ring XY-mixer, then a classical integer-program allocator on the chosen subset).
5. `coskweness_cokurtosis.py` — functions to compute the higher-order moments.
6. `portfolio_higher_moments_classical.py` — the classical baselines.
7. `profile.py` — static circuit specs and convergence profiling for either/both methods.

This project is managed by **uv** — run everything with `uv run python ...` (it auto-syncs the environment first).

## Running the experiments

`experiments.py` takes `batch_num` and `total_batches` (for splitting the 100 problems across SLURM array jobs) plus `--method {hopo,ring_xy}`. The shared
classical and exact baselines are computed and written for either method, so each
output JSON is self-contained.

### Original pipeline (HOPO + raw QAOA)

```bash
# Local single batch (batch 0 of 1 = all 100 problems)
uv run python experiments.py 0 1 --method hopo

# SLURM array (100 parallel jobs)
sbatch run.sh
```

Writes `portfolio_optimization_batch_<optimizer>_<lambda>_<batch>.json`
(e.g. `portfolio_optimization_batch_CMAES_0.001_0.json`). Each entry contains
`hyperparams`, `continuous_variables_solution`,
`continuous_variables_solution_unconstrained`, `exact_solution`, and
`qaoa_solution`.

### New pipeline (ring XY-mixer cardinality selection)

```bash
# Local single batch
uv run python experiments.py 0 1 --method ring_xy

# SLURM array (100 parallel jobs)
sbatch run_ring_xy.sh
```

Writes `ring_xy_batch_<batch>.json` (e.g. `ring_xy_batch_0.json`). Each entry
contains the same shared baselines plus `cardinality_qaoa_solution`, which sweeps
K = 2..N and records `per_K` results and the `best_K` (the K with the best
post-allocation objective). It does **not** contain `qaoa_solution`.

Both runners save results incrementally after each problem, so an interrupted job
can be resumed.

## Comparing circuit specs with profile.py

`profile.py specs` reports static circuit metrics (qubit count, depth, gate
counts, two-qubit-gate count) with no QAOA optimization, so it is cheap. Use
`--method` to choose which circuits to profile:

```bash
# Profile both methods side by side on every problem
uv run python profile.py specs --method both --out profile_results.json

# Or one method at a time
uv run python profile.py specs --method hopo
uv run python profile.py specs --method ring_xy
```

Useful flags: `--max-problems N` (limit how many problems), `--sweep` (profile
every K = 2..N for the cardinality circuit instead of just `best_K`),
`--rederive-k` (recompute `best_K` by running the K sweep instead of reading it
from the latest `ring_xy_batch_*.json`).

### What to expect

`profile_results.json` has a `static_specs` list with one row per circuit. Each
row carries a `method` field — `"integer_hubo"` (original) or `"cardinality"`
(ring XY) — so the two can be compared directly. With `--method both` you get
both row types per problem; with a single method you get only that type.

Typical differences for the same problem:

- **Qubit count** — `integer_hubo` uses **log-encoded** qubits (multiple qubits
  per asset, sized to the budget), so its `n_qubits` is usually larger than the
  asset count. `cardinality` uses **one qubit per asset** (`n_qubits == n_assets`),
  so it is generally smaller.
- **Initial state / mixer** — `integer_hubo` is Hadamard-init + X-mixer (no state
  preparation gates). `cardinality` prepares a Dicke state and uses a ring
  XY-mixer, so its rows show a small number of `RY` gates (these appear under
  `unexpected_gates: ["RY"]`, which is expected for this method — the field just
  flags gates outside the `CNOT/RZ/RX/Hadamard` target set).
- **Depth / two-qubit gates** — both grow with the number of degree-3/4 HUBO
  terms (from coskewness/cokurtosis), so larger problems show higher `depth` and
  `two_qubit_gate_count` for both methods.

Note: `depth`, gate counts, and `unexpected_gates` ordering can differ slightly
between separate runs of the same problem because Python's per-process hash
seed affects HUBO term ordering (and thus downstream gate cancellation during
compilation). Fix `PYTHONHASHSEED` if you need bit-stable spec numbers across
runs.

To also measure optimizer dynamics (CMA-ES evaluations, wall-clock, and the
approximation ratio against the exact spectrum), use the more expensive
`convergence` subcommand:

```bash
uv run python profile.py convergence --method both --per-qubit 1
```
