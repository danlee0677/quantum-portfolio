"""Confirm the current HOPO-with-QAOA pipeline still reproduces the author's
saved results, before trusting the migrated baseline that
``profile.py --hopo-source json`` ingests.

This is a *lightweight* drift check meant to run on 1-2 small experiments on the
user's own hardware -- not a full reproduction. For each requested experiment it
rebuilds the exact problem from the author's stored hyperparams (so it does not
depend on experiments_data.json matching), runs the current
``HigherOrderPortfolioQAOA`` with the author's hyperparameters
(``lambda_budget=1.0`` = the canonical ``lambda_1`` baseline, CMAES,
risk_aversion 0.1), and diffs against the author's entry in three tiers:

  * DATA PARITY   -- fresh yfinance prices_now vs the stored prices_now. A
                     divergence means downstream diffs are data-driven (yfinance
                     revisions), not code drift -> reported, never a FAIL.
  * DETERMINISTIC -- exact eigen-spectrum + continuous baseline. These ignore
                     CMA-ES stochasticity and MUST match -> a mismatch FAILs
                     (the pipeline drifted; do not trust the migrated baseline).
  * QAOA BAND     -- CMA-ES is not seeded, so QAOA values vary run-to-run. Run a
                     few times and check the author's value lands in a sane band.
                     Informational, never a hard FAIL on a small sample.

The reference defaults to the newest migrated current-format file in results/;
pass --reference to point elsewhere. Exits non-zero iff a deterministic tier
fails on any requested id.

Usage:
    uv run python verify_hopo_equivalence.py <id> [<id> ...]
    uv run python verify_hopo_equivalence.py 0 1 --qaoa-runs 3
"""

import argparse
import sys

import numpy as np
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov

from coskweness_cokurtosis import coskewness, cokurtosis
from portfolio_hubo_qaoa_light import HigherOrderPortfolioQAOA
from profile import build_hopo_results, prepare_problem_data

# Tolerances.
TOL_DATA = 1e-3      # relative; yfinance revisions allowed below this
TOL_EIG = 1e-6       # relative; eigendecomposition must reproduce
TOL_CONT = 1e-4      # relative; continuous (convex) objective value
TOL_AR_BAND = 0.05   # absolute; approximation-ratio band for the stochastic QAOA


def rel_diff(a, b):
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def rel_close(a, b, tol):
    return rel_diff(float(a), float(b)) <= tol


def load_reference(ids, reference):
    """Author's per-id entries, from an explicit file or the merged results/ glob."""
    id_range = (min(ids), max(ids))
    merged, _, files = build_hopo_results(id_range, reference)
    if files:
        print("Reference file(s):")
        for f in files:
            print(f"  {f}")
    return merged


def build_hubo_from_reference(entry, lambda_budget):
    """Rebuild the exact problem the author solved, from their stored hyperparams,
    and return (portfolio_hubo, fresh_prices_now, stored_prices_now)."""
    hp = entry["hyperparams"]
    experiment = {
        "stocks": list(hp["stocks"]),
        "start": hp["start"],
        "end": hp["end"],
        "budget": hp["budget"],
    }
    data = prepare_problem_data(experiment)
    fresh_prices = {str(k): float(v) for k, v in data["prices_now"].items()}
    hubo = HigherOrderPortfolioQAOA(
        stocks=data["stocks"],
        prices_now=data["prices_now"],
        expected_returns=data["expected_returns"],
        covariance_matrix=data["covariance_matrix"],
        budget=data["budget"],
        max_qubits=15,
        coskewness_tensor=data["coskewness_tensor"],
        cokurtosis_tensor=data["cokurtosis_tensor"],
        log_encoding=True,
        risk_aversion=hp.get("risk_aversion", 0.1),
        strict_budget_constraint=False,
        lambda_budget=lambda_budget,
    )
    return hubo, fresh_prices, dict(hp["prices_now"])


def check_data_parity(fresh_prices, stored_prices):
    notes = []
    ok = True
    for tic, stored in stored_prices.items():
        if tic not in fresh_prices:
            notes.append(f"    {tic}: missing in fresh download")
            ok = False
            continue
        if not rel_close(fresh_prices[tic], stored, TOL_DATA):
            notes.append(f"    {tic}: fresh={fresh_prices[tic]:.6g} stored={stored:.6g} "
                         f"(rel {rel_diff(fresh_prices[tic], stored):.2e})")
            ok = False
    return ok, notes


def check_deterministic(hubo, entry):
    """Returns (passed, lines). FAIL on any eigen/continuous mismatch."""
    lines = []
    passed = True
    ref_exact = entry.get("exact_solution") or {}

    # Exact eigen-spectrum.
    try:
        (smallest_eigs, _, _, opt_portfolio, _, spectrum, _, _) = hubo.solve_exactly()
    except Exception as e:
        return False, [f"    solve_exactly failed: {e}"]

    ref_eig0 = ref_exact.get("smallest_eigenvalues", [None])[0]
    if ref_eig0 is not None:
        ok = rel_close(smallest_eigs[0], ref_eig0, TOL_EIG)
        passed &= ok
        lines.append(f"    min eigenvalue: new={float(smallest_eigs[0]):.10g} "
                     f"ref={float(ref_eig0):.10g}  {'OK' if ok else 'MISMATCH'}")

    # Old ≥14-qubit references stored a *truncated* spectrum -- 3 elements from the
    # eigsh(k=3) path, or 2 from the solve_exactly_with_lobpcg(k=2) fallback. We
    # detect that self-describingly as "shorter than the live full diagonal", which
    # covers both producers and auto-stops once references are regenerated in the
    # new full-spectrum format (len == 2**n == len(spectrum)). For such references
    # two fields legitimately disagree with the corrected diagonal solver and must
    # NOT count as drift:
    #   * spectrum max  -- a truncated spectrum's "max" is the 2nd/3rd-smallest
    #     eigenvalue, not the true maximum.
    #   * optimized_portfolio -- the old eigsh path one-hots via argmax on an
    #     ARPACK eigenvector (defined up to sign), which could store a wrong ground
    #     portfolio; for lobpcg-origin references a mismatch would instead be
    #     degeneracy. Either way it is not new code drift.
    # The min-eigenvalue and spectrum-min checks stay STRICT: the stored eigenvalues
    # were correct even when the eigenvector sign tripped the portfolio bug.
    ref_spectrum = ref_exact.get("spectrum")
    old_format = isinstance(ref_spectrum, list) and 0 < len(ref_spectrum) < len(spectrum)
    if ref_spectrum:
        ok = rel_close(min(spectrum), min(ref_spectrum), TOL_EIG)
        passed &= ok
        lines.append(f"    spectrum min: new={float(min(spectrum)):.10g} "
                     f"ref={float(min(ref_spectrum)):.10g}  {'OK' if ok else 'MISMATCH'}")
        ok = rel_close(max(spectrum), max(ref_spectrum), TOL_EIG)
        if old_format and not ok:
            lines.append(f"    spectrum max: new={float(max(spectrum)):.10g} "
                         f"ref={float(max(ref_spectrum)):.10g}  expected-mismatch "
                         f"(old 3-element spectrum, not drift)")
        else:
            passed &= ok
            lines.append(f"    spectrum max: new={float(max(spectrum)):.10g} "
                         f"ref={float(max(ref_spectrum)):.10g}  {'OK' if ok else 'MISMATCH'}")

    ref_opt = ref_exact.get("optimized_portfolio")
    if ref_opt is not None:
        ok = (opt_portfolio == ref_opt)
        if old_format and not ok:
            lines.append(f"    exact optimized_portfolio: expected-mismatch "
                         f"(truncated old-format reference, not drift)  "
                         f"new={opt_portfolio} ref={ref_opt}")
        else:
            passed &= ok
            lines.append(f"    exact optimized_portfolio: {'OK' if ok else 'MISMATCH'}"
                         + ("" if ok else f"  new={opt_portfolio} ref={ref_opt}"))

    # Continuous (constrained) baseline.
    ref_cont = entry.get("continuous_variables_solution") or {}
    try:
        _, allocation, value, _ = hubo.solve_with_continuous_variables()
        if "value" in ref_cont:
            ok = rel_close(value, ref_cont["value"], TOL_CONT)
            passed &= ok
            lines.append(f"    continuous value: new={float(value):.10g} "
                         f"ref={float(ref_cont['value']):.10g}  {'OK' if ok else 'MISMATCH'}")
        if "allocation" in ref_cont:
            ok = (allocation == ref_cont["allocation"])
            # Allocation can shift on tiny weight changes; report but do not FAIL.
            lines.append(f"    continuous allocation: {'OK' if ok else 'differs (informational)'}"
                         + ("" if ok else f"  new={allocation} ref={ref_cont['allocation']}"))
    except Exception as e:
        lines.append(f"    solve_with_continuous_variables failed: {e}")
        passed = False

    return passed, lines


def check_qaoa_band(hubo, entry, runs):
    ref_qaoa = entry.get("qaoa_solution") or {}
    ref_fev = ref_qaoa.get("final_expectation_value")
    ref_exact = entry.get("exact_solution") or {}
    spectrum = ref_exact.get("spectrum")
    E_min = min(spectrum) if spectrum else None
    E_max = max(spectrum) if spectrum else None

    fevs = []
    for _ in range(runs):
        try:
            res = hubo.solve_with_qaoa_cma_es()
            fevs.append(float(res[1]))
        except Exception as e:
            return [f"    QAOA run failed: {e}"]
    lines = [f"    local final_expectation_value over {runs} run(s): "
             f"min={min(fevs):.6g} max={max(fevs):.6g} mean={np.mean(fevs):.6g}"]
    if ref_fev is not None:
        ref_fev = float(ref_fev)
        in_local = min(fevs) - 1e-9 <= ref_fev <= max(fevs) + 1e-9
        lines.append(f"    author fev={ref_fev:.6g} "
                     f"{'within' if in_local else 'OUTSIDE'} local range")
        if E_min is not None and E_max is not None and E_max > E_min:
            ref_ar = (ref_fev - E_min) / (E_max - E_min)
            local_ars = [(f - E_min) / (E_max - E_min) for f in fevs]
            near = any(abs(ref_ar - a) <= TOL_AR_BAND for a in local_ars)
            lines.append(f"    author approx-ratio={ref_ar:.4f}; local="
                         f"[{min(local_ars):.4f},{max(local_ars):.4f}]; "
                         f"{'in band' if near else 'out of band'} (+/-{TOL_AR_BAND})")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ids", type=int, nargs="+", help="Experiment id(s) to verify.")
    parser.add_argument("--reference", default=None,
                        help="Explicit author/current-format results file "
                             "(default: newest results/portfolio_optimization_exp_*.json).")
    parser.add_argument("--lambda-budget", type=float, default=1.0,
                        help="Budget penalty to match the reference (default 1.0 = lambda_1).")
    parser.add_argument("--qaoa-runs", type=int, default=2,
                        help="Local QAOA repeats for the stochastic band (default 2; 0 to skip).")
    args = parser.parse_args()

    reference = load_reference(args.ids, args.reference)
    if not reference:
        print("No reference entries found; nothing to verify.")
        sys.exit(2)

    any_fail = False
    any_inconclusive = False
    for pid in args.ids:
        entry = reference.get(str(pid))
        print(f"\n=== experiment {pid} ===")
        if entry is None or "hyperparams" not in entry:
            print("  no usable reference entry; skipping")
            continue

        hubo, fresh_prices, stored_prices = build_hubo_from_reference(entry, args.lambda_budget)

        data_ok, data_notes = check_data_parity(fresh_prices, stored_prices)
        print(f"  DATA PARITY: {'OK' if data_ok else 'DIVERGED (data-driven)'}")
        for n in data_notes:
            print(n)

        det_ok, det_lines = check_deterministic(hubo, entry)
        for line in det_lines:
            print(line)
        # A deterministic mismatch is only attributable to *code* drift when the
        # input data matched. yfinance re-adjusts historical prices over time, so
        # diverged data makes the comparison inconclusive (the author's exact
        # returns are not recoverable -- which is precisely why the migrated JSON
        # is the only faithful baseline).
        if det_ok:
            print("  DETERMINISTIC: PASS")
        elif data_ok:
            print("  DETERMINISTIC: FAIL (code drift -- data matched but results differ)")
            any_fail = True
        else:
            print("  DETERMINISTIC: INCONCLUSIVE (data diverged; diffs are data-driven, "
                  "not code drift)")
            any_inconclusive = True

        if args.qaoa_runs > 0:
            print("  QAOA BAND (informational):")
            for line in check_qaoa_band(hubo, entry, args.qaoa_runs):
                print(line)

    print()
    if any_fail:
        print("VERDICT: FAIL (deterministic code drift detected)")
        sys.exit(1)
    if any_inconclusive:
        print("VERDICT: INCONCLUSIVE (input data has drifted vs the author's download; "
              "cannot confirm/deny code parity -- ingest the migrated JSON as the baseline)")
        sys.exit(0)
    print("VERDICT: PASS (deterministic parity holds)")
    sys.exit(0)


if __name__ == "__main__":
    main()
