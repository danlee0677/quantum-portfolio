## Main files in this repo

1. The notebook `generate_portfolio_experiments.ipynb` generates random portfolio optimization problems based on the real stock data from yfinance, written to `experiments_data.json`.
2. `experiments.py` — the main experiment runner. Selects one of two QAOA algorithms via `--method`.
3. `portfolio_hubo_qaoa_light.py` — class `HigherOrderPortfolioQAOA`, the **original paper method** (integer-HUBO + raw QAOA, X-mixer, budget penalty). Also holds the shared classical/exact baselines.
4. `ring_xy.py` — class `RingXYCardinalityQAOA`, the **new cardinality-selection method** (picks K of N stocks with a Dicke initial state + ring XY-mixer, then a classical integer-program allocator on the chosen subset). The cardinality K is chosen by a cheap classical selector (`lasso_localize_K` + `select_K_classically`) so the expensive fixed-K QAOA runs only on the top candidate(s).
5. `coskweness_cokurtosis.py` — functions to compute the higher-order moments.
6. `portfolio_higher_moments_classical.py` — the classical baselines.
7. `profile.py` — static circuit specs and convergence profiling for either/both methods. Either side of the convergence comparison can be ingested from saved results instead of recomputed (`convergence --hopo-source json` / `--card-source json`).
8. `migrate_author_hopo_results.py` — converts the previous author's HOPO results (`results_old/cmaes_hubo_results/lambda_<L>/`) into the current pipeline's JSON format so `profile.py` can ingest them.
9. `verify_hopo_equivalence.py` — lightweight check that the current pipeline still reproduces the author's saved results (distinguishes code drift from yfinance data drift).

This project is managed by **uv** — run everything with `uv run python ...` (it auto-syncs the environment first).

## Running the experiments

`experiments.py` takes an inclusive, **0-based experiment range** `start end`
(e.g. `1 4` runs experiments #1, #2, #3, #4) plus `--method {hopo,ring_xy}`. The
shared classical and exact baselines are computed and written for either method,
so each output JSON is self-contained.

All results are written to `results/`. Each run gets its own filename that embeds
the experiment range **and** a `YYYYMMDD_HHMMSS` timestamp, so re-running a range
never overwrites an earlier run — older files are kept side by side.

### Original pipeline (HOPO + raw QAOA)

```bash
# Run experiments 0 through 99 (all 100 problems)
uv run python experiments.py 0 99 --method hopo

# Run just experiments 1 through 4
uv run python experiments.py 1 4 --method hopo
```

Writes `results/portfolio_optimization_exp_<optimizer>_<lambda>_<start>_<end>_<timestamp>.json`
(e.g. `results/portfolio_optimization_exp_CMAES_0.001_1_4_20260610_153000.json`).
Each entry contains `hyperparams`, `continuous_variables_solution`,
`continuous_variables_solution_unconstrained`, `exact_solution`, and
`qaoa_solution`.

### New pipeline (ring XY-mixer cardinality selection)

```bash
# Run experiments 1 through 4 (default: classical LASSO K-selector)
uv run python experiments.py 1 4 --method ring_xy
```

Writes `results/ring_xy_exp_<start>_<end>_<timestamp>.json`
(e.g. `results/ring_xy_exp_1_4_20260610_153000.json`). Each entry contains the
same shared baselines plus `cardinality_qaoa_solution`, with the `best_K` (the K
with the best post-allocation objective) and `per_K` results. It does **not**
contain `qaoa_solution`.

**Choosing the cardinality K.** The number of assets K is no longer brute-forced.
By default a cheap, purely-classical selector picks K so the expensive fixed-K
QAOA runs only on the top candidate(s):

1. **LASSO localizer** — an L1-penalized continuous higher-moment solve (a short
   λ-path) gives a ballpark `K̂` and an asset ranking (`lasso_localize_K`).
2. **Classical neighborhood scan** — K in `[K̂−w, K̂+w]` is scored by allocating
   the top-K assets with the existing weighted objective + discrete allocator
   (`select_K_classically`). No QAOA and no subset enumeration, so it scales to
   large N (1000+), unlike the old K = 2..N sweep.
3. **Fixed-K QAOA** — runs only on the top `--k-hedge` classical K; `best_K` is
   the QAOA-run K with the best post-allocation objective (same rule as before).

Relevant flags (all `ring_xy` only):

- `--k-selector {lasso,sweep,both}` (default `lasso`) — `sweep` reproduces the
  original brute force over K = 2..N; `both` runs the full sweep **and** records
  what `lasso` would have picked (`lasso_picked_K`) for offline accuracy checks.
- `--k-neighborhood N` (default 2) — half-width of the classical K scan around `K̂`.
- `--k-hedge N` (default 2) — how many top classical K to actually run QAOA on.
- `--k-min N` (default 2) — smallest K considered.

```bash
# Reproduce the original brute-force K sweep
uv run python experiments.py 1 4 --method ring_xy --k-selector sweep

# Validation: full sweep ground truth + record the lasso pick on the same problem
uv run python experiments.py 1 4 --method ring_xy --k-selector both
```

Beyond `best_K` and `per_K`, `cardinality_qaoa_solution` records the selector
trail: `k_selector`, `qaoa_K_run` (the K actually given to QAOA), and — for the
`lasso`/`both` selectors — `k_hat`, `classical_K_scan` (the cheap per-K classical
curve), `lasso_support_path`, and (in `both` mode) `lasso_picked_K`. With the
default `lasso` selector, `per_K` holds only the hedged K (≤ `--k-hedge` entries)
instead of the full sweep.

Both runners save results incrementally after each problem, so an interrupted job
can be resumed: a later run skips any experiment id already present in an earlier
`results/` file for the same method. (This is why re-running a range that you've
already partly computed appears to "start in the middle" — the earlier ids are
skipped. To force a clean re-run, move or delete the older `results/` files for
that method first.)

**Exact baseline on large encodings (≥14 qubits).** Both methods write a shared
`exact_solution` baseline — the true global optimum, used as the common reference
for cross-method comparison (e.g. the approximation ratio). It is computed by
diagonalizing the *integer-HUBO* cost Hamiltonian, whose width grows with the
budget: even a 2-stock problem can need ≥14 qubits because share counts are
log-encoded (this is the integer-HUBO baseline, **not** the ring XY-mixer circuit,
which stays one qubit per asset). `solve_exactly` exploits the fact that this cost
Hamiltonian is **diagonal**, so these large cases now resolve in well under a
second — an earlier version effectively hung on them by running a slow sparse
eigensolver (`eigsh(..., which='SA')`). For ≥14-qubit experiments,
`exact_solution.spectrum` now stores the full 2ⁿ spectrum (~0.5–0.7 MB per such
experiment), so `E_min`/`E_max` and approximation ratios are exact. **Results
produced before this fix at ≥14 qubits — including any migrated author JSON —
carry a wrong ground `optimized_portfolio` (an ARPACK eigenvector-sign bug) and a
truncated 3-value spectrum; regenerate them.** `verify_hopo_equivalence.py` treats
such old-format mismatches (`optimized_portfolio`, spectrum max) as expected, not
code drift.

**Unavailable / delisted tickers.** Stock symbols are resolved live from yfinance,
so a ticker that has since been delisted or renamed (e.g. `WBA`, taken private in
2025) comes back as an all-NaN column. The runner now **drops such tickers** after
download and logs them; if fewer than 2 valid tickers remain for a problem, that
experiment is **skipped** and recorded as `{"error": "insufficient_valid_tickers",
"dropped_tickers": [...], "requested_stocks": [...]}` instead of crashing the
batch. Note that dropping a ticker changes that problem from its original
definition (fewer assets than `n_stocks`), so affected experiments are no longer
directly comparable to the original paper's problem set — check `dropped_tickers`
in the output if a result looks off.

> **Note:** the SLURM scripts (`run.sh`, `run_ring_xy.sh`) still pass the old
> `batch_num total_batches` arguments and need updating to the `start end` range
> signature before they will run.

## Comparing circuit specs with profile.py

`profile.py specs` reports static circuit metrics (qubit count, depth, gate
counts, two-qubit-gate count) with no QAOA optimization, so it is cheap. Use
`--method` to choose which circuits to profile, and an optional **0-based
inclusive `start end` range** (matching `experiments.py`) to restrict which
experiments are profiled:

```bash
# Profile both methods side by side on every problem
uv run python profile.py specs --method both --out profile_results.json

# Profile only experiments 1 through 4
uv run python profile.py specs 1 4 --method ring_xy

# Or one method at a time, all problems
uv run python profile.py specs --method hopo
```

(Omit `start end` to profile all problems. A single value, e.g. `profile.py
specs 3`, profiles just experiment #3.)

**Where `best_K` comes from.** For the cardinality method, `profile.py` reads
`cardinality_qaoa_solution.best_K` from the `results/` files. Rather than picking
a single file, it assembles the data **per experiment from the newest file that
contains it**: if you request `1 4` and have an older `ring_xy_exp_1_3_*.json`
plus a newer `ring_xy_exp_2_4_*.json`, it pulls experiments 2–4 from the newer
file and experiment 1 from the older one. The files actually used and a
per-experiment provenance map are recorded in the output's `meta` block and
printed at startup.

Useful flags: `--results-json PATH` (bypass the search and read `best_K` from one
specific file), `--max-problems N` (limit how many problems), `--sweep` (profile
every K = 2..N for the cardinality circuit instead of just `best_K`),
`--rederive-k` (recompute `best_K` by running the K sweep instead of reading it
from `results/`).

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
  XY-mixer; the Dicke `StatePrep` compiles (via Möttönen state preparation) into
  `RY` + `CNOT` gates, and each surviving `RY` is then rewritten as
  `RZ(−π/2)·RX·RZ(π/2)` so both methods report resources in the identical
  `CNOT/RZ/RX/Hadamard` basis. `unexpected_gates` is expected to be empty for
  both methods (spec files generated before this rewrite still carry
  `unexpected_gates: ["RY"]`).
- **Depth / two-qubit gates** — both grow with the number of degree-3/4 HUBO
  terms (from coskewness/cokurtosis), so larger problems show higher `depth` and
  `two_qubit_gate_count` for both methods.

Note: `depth`, gate counts, and `unexpected_gates` ordering can differ slightly
between separate runs of the same problem because Python's per-process hash
seed affects HUBO term ordering (and thus downstream gate cancellation during
compilation). Fix `PYTHONHASHSEED` if you need bit-stable spec numbers across
runs.

## Measuring optimizer dynamics with `profile.py convergence`

To also measure optimizer dynamics (CMA-ES evaluations, wall-clock, and the
approximation ratio against the exact spectrum), use the more expensive
`convergence` subcommand:

```bash
uv run python profile.py convergence --method both --per-qubit 1

# The same start end range works here too
uv run python profile.py convergence 1 4 --method ring_xy
```

**Qubit buckets (`--per-qubit`, default 1).** Because a live convergence run is
expensive, problems are sampled per qubit count rather than run exhaustively: the
first `--per-qubit` problems encountered at each qubit count are profiled and the
rest are skipped. So `convergence 0 99 --per-qubit 1` gives one problem per qubit
size (the dataset spans 6–15 integer-HUBO qubits), and raising `--per-qubit`
widens each bucket. The bucketing key depends on `--method`: with `hopo` or
`both`, problems are bucketed by the **integer-HUBO qubit count**
(`hyperparams.n_qubits`); with `ring_xy` alone, by the **cardinality circuit's
qubit count**, which is just the number of assets. The same value is written to
each row's `n_qubits` field.

**What a row contains.** Output goes to `--out` (default
`results_profile_profile_results_<timestamp>.json`) as a `convergence` list with
one row per profiled circuit, tagged `method: "integer_hubo"` or
`"cardinality"` and `source: "live"` or `"json"` (see below). Both row types
carry `wall_clock_seconds`, `evaluations` / `iterations` (CMA-ES effort),
`final_expectation_value`, `post_objective`, and the full `training_history`.
Integer-HUBO rows add `approximation_ratio` with `E_min`/`E_max` from the exact
spectrum (min–max convention, 0 = optimal); cardinality rows add `K`, both
`approximation_ratio_subspace` (1 = optimal) and `approximation_ratio_global`
(HOPO-parity min–max, 0 = optimal) with their `E_*` references, and
`infeasible`. The cardinality rows cover only `best_K` by default — `--sweep`
profiles every K = 2..N instead, and `--rederive-k` recomputes `best_K` live
rather than reading it from `results/` (both force live runs).

## Loading saved results instead of recomputing (`--hopo-source` / `--card-source`)

By default `convergence` recomputes **both** methods live on every run — the
expensive, stochastic CMA-ES optimizations — so numbers jitter run to run.
Each side can instead be loaded from the saved `results/` JSON, giving fixed,
reproducible rows:

```bash
# Load the HOPO baseline from JSON; ring_xy is still computed live
uv run python profile.py convergence 0 4 --method both --hopo-source json

# Load both sides from JSON: no QAOA, no yfinance download — runs in seconds
uv run python profile.py convergence 0 4 --method both --hopo-source json --card-source json
```

**`--hopo-source {live,json}`** (default `live`) — `json` loads the integer-HUBO
row from saved `results/portfolio_optimization_exp_*.json` instead of re-running
`solve_with_qaoa_cma_es` + `solve_exactly`:

- `final_expectation_value` / `post_objective` / `training_history` come from the
  saved `qaoa_solution`; `E_min`/`E_max` (for the approximation ratio) from
  `exact_solution.spectrum`.
- Files are merged **per experiment from the newest one that contains it** (same
  newest-first logic as `best_K`), independent of the ring_xy stream.
  `--hopo-results-json PATH` forces a single file.

**`--card-source {live,json}`** (default `live`) — the mirror for the
cardinality side: `json` loads the `best_K` entry from
`cardinality_qaoa_solution.per_K` in the saved `results/ring_xy_exp_*.json`
instead of re-running `solve_with_qaoa_cardinality(best_K)`:

- All metrics (`final_expectation_value`, both approximation ratios, the
  `E_*` energy references, `post_objective`, `infeasible`, `training_history`)
  are copied verbatim from the saved per-K entry, so they are
  convention-identical to live rows.
- It reads the **same merged ring_xy stream** already used for `best_K`
  resolution (`--results-json PATH` forces a single file). The `best_K` entry is
  always present in `per_K`, even when the default lasso selector truncates
  `per_K` to the hedged K candidates.
- `--sweep` and `--rederive-k` explicitly request recomputation, so they bypass
  the JSON path (a note is printed).

Shared behavior for both flags:

- JSON-loaded rows are tagged `"source": "json"` and have
  `wall_clock_seconds: null` (the timing is not recoverable from the saved
  files); live rows are tagged `"source": "live"`.
- Any experiment missing from the JSON (or with an unusable entry) **falls back
  to a live recompute** for just that experiment, with a printed note.
- A pure-JSON run (`--method hopo --hopo-source json`, `--method ring_xy
  --card-source json`, or `--method both` with both flags) skips the yfinance
  download and all solver builds entirely.

## Plotting the profiles with `profile_visualize.py`

`profile_visualize.py` turns the JSON written by `profile.py` into comparison
figures (and CSV summaries) contrasting the original `integer_hubo` pipeline with
the ring-XY `cardinality` contribution. It does **no file discovery** — you pass
the exact files to plot, and it prints which file it is using for each.

It takes two independent path flags, one per `profile.py` subcommand:

- `--specs PATH` — a `profile.py specs` JSON; draws circuit-resource scaling,
  per-problem savings, and the gate-composition figure (plus `specs_summary.csv`).
- `--convergence PATH` — a `profile.py convergence` JSON; draws the convergence
  overview, per-problem savings / quality face-off (plus `convergence_summary.csv`).

**At least one of the two must be given** — the program refuses to run with no
input file. Pass both to render every figure in one go. Each file is checked
against its expected `meta.subcommand`, so a specs file handed to `--convergence`
(or vice-versa) is rejected.

```bash
# Plot just the specs profile
uv run python profile_visualize.py --specs profile_results.json

# Plot just the convergence profile
uv run python profile_visualize.py --convergence results_profile_..._20260610.json

# Plot both at once, into a chosen output directory
uv run python profile_visualize.py \
    --specs specs_profile.json \
    --convergence convergence_profile.json \
    --outdir figures
```

Figures are written to `--outdir` (default `figures/`) as headless PNGs; pass
`--show` to display them interactively instead.

### Migrating the previous author's results

The earlier author's HOPO results live in `results_old/cmaes_hubo_results/lambda_<L>/`
(one `portfolio_optimization_results_batch_*.json` per experiment; `lambda_1` is
the canonical `lambda_budget = 1.0` baseline). Convert them into one current-format
file in `results/` so `--hopo-source json` can read them:

```bash
# <stamp> is a YYYYMMDD_HHMMSS token used for newest-first sorting
uv run python migrate_author_hopo_results.py 20260610_120000
# -> results/portfolio_optimization_exp_CMAES_1.0_0_99_<stamp>.json
```

Migration just merges the batch files and adds the two fields the current pipeline
writes (`hyperparams.{optimizer,lambda_budget}`,
`continuous_variables_solution_unconstrained = null`); the deterministic baselines
are left untouched. Use `--lambda-folder lambda_<L>` to migrate a different budget
penalty.

### Verifying the pipeline hasn't drifted

Before trusting that baseline, `verify_hopo_equivalence.py` rebuilds 1–2 problems
from the author's stored hyperparameters, runs the current pipeline at
`lambda_budget = 1.0`, and diffs three tiers: **data parity** (fresh yfinance
`prices_now` vs stored), **deterministic** (exact eigen-spectrum + continuous
baseline — must match), and a **QAOA band** (CMA-ES is unseeded, so only checked
for plausibility).

```bash
uv run python verify_hopo_equivalence.py 0 1 --qaoa-runs 2
```

A deterministic mismatch counts as a code-drift **FAIL only when the input data
matched**. In practice yfinance **re-adjusts historical close prices over time**,
so a fresh download no longer matches the author's, and the verdict is
**INCONCLUSIVE** — the author's exact returns are not recoverable, which is exactly
why the migrated JSON is the only faithful baseline (you cannot regenerate those
numbers locally). The script exits non-zero only on a genuine code-drift FAIL.
