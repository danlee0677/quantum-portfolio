"""Visualize ``profile_results.json`` produced by ``profile.py``.

Dispatches on ``meta["subcommand"]`` and draws method-comparison figures
contrasting the original-paper ``integer_hubo`` pipeline with the ring-XY
``cardinality`` contribution.

  uv run python profile_visualize.py [--specs PATH] [--convergence PATH] \
      [--outdir DIR] [--show]

At least one of ``--specs`` / ``--convergence`` must be given (the program
refuses to run with no input file) and the file actually used is printed.

* ``--specs``       -> circuit-resource scaling, per-problem savings, gate mix.
* ``--convergence`` -> solve quality / cost scatter + per-problem savings.

Static specs depend only on circuit *shape*; convergence rows are CMA-ES
end-state scalars (no per-iteration trace exists), so convergence plots are
scatter/box summaries, not training curves.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib


# Fixed per-method styling reused across every figure so the two methods read
# consistently. integer_hubo = original HOPO paper; cardinality = ring-XY.
METHOD_STYLE = {
    "integer_hubo": {"color": "#c44e52", "marker": "o", "label": "integer-HUBO (HOPO)"},
    "cardinality": {"color": "#4c72b0", "marker": "s", "label": "cardinality (ring-XY)"},
}
METHOD_ORDER = ["integer_hubo", "cardinality"]

# Extra series for the Fig. 6 comparison (objective vs budget utilization). The
# two QAOA methods reuse METHOD_STYLE; continuous/reference get their own styling,
# and over-budget integer-HUBO points (budget_util > 1) are segregated.
COMPARE_STYLE = {
    "integer_hubo": METHOD_STYLE["integer_hubo"],
    "cardinality": METHOD_STYLE["cardinality"],
    "continuous": {"color": "#000000", "marker": "X",
                   "label": "continuous + rounding (baseline)"},
    "continuous_unconstrained": {"color": "#7f7f7f", "marker": "P",
                                 "label": "continuous (unconstrained)"},
    "reference": {"color": "#55a868", "marker": "*",
                  "label": "exact reference (allocator-optimal)"},
    "over_budget": {"color": "#c44e52", "marker": "x",
                    "label": "integer-HUBO over-budget (util > 1)"},
}
# budget_util <= 1 + tol counts as feasible (mirrors profile.BUDGET_TOL).
COMPARE_BUDGET_TOL = 1e-6

# Gate types charted in the composition figure (union across both methods).
# RY is kept for backward compatibility: spec files generated before the
# RY -> RZ·RX·RZ rewrite in profile.py's compile_for_specs still contain it.
GATE_TYPES = ["CNOT", "RZ", "RX", "RY", "Hadamard"]
GATE_COLORS = {
    "CNOT": "#c44e52", "RZ": "#4c72b0", "RX": "#55a868",
    "RY": "#8172b3", "Hadamard": "#ccb974",
}


# --------------------------------------------------------------------------- #
# data helpers
# --------------------------------------------------------------------------- #
def _ok(row, *fields):
    """True if ``row`` is usable and every field is present and non-None."""
    if not isinstance(row, dict) or "error" in row:
        return False
    return all(row.get(f) is not None for f in fields)


def _finite(x):
    """True for a real, finite number (rejects None / NaN / inf / non-numeric)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x)


def group_by_method(rows):
    """Split usable rows by method, preserving METHOD_ORDER."""
    out = {m: [] for m in METHOD_ORDER}
    for r in rows:
        m = r.get("method")
        if m in out and "error" not in r:
            out[m].append(r)
    return out


def median_curve(rows, x, y):
    """Return ``(xs, medians)`` sorted by x, dropping rows missing x or y."""
    buckets = defaultdict(list)
    for r in rows:
        if _ok(r, x, y):
            buckets[r[x]].append(r[y])
    xs = sorted(buckets)
    return xs, [float(np.median(buckets[k])) for k in xs]


def paired_by_problem(rows, reduce_field):
    """Map problem_id -> {method: row}, keeping only problems present for both
    methods. If a method has several rows for one problem (e.g. cardinality K
    sweep), collapse to the row whose ``reduce_field`` is the median."""
    by_pid = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if "error" in r or r.get("method") not in METHOD_ORDER:
            continue
        by_pid[r.get("problem_id")][r["method"]].append(r)

    paired = {}
    for pid, methods in by_pid.items():
        if not all(methods.get(m) for m in METHOD_ORDER):
            continue
        chosen = {}
        ok = True
        for m in METHOD_ORDER:
            cands = [r for r in methods[m] if _ok(r, reduce_field)]
            if not cands:
                ok = False
                break
            vals = [c[reduce_field] for c in cands]
            target = float(np.median(vals))
            chosen[m] = min(cands, key=lambda c: abs(c[reduce_field] - target))
        if ok:
            paired[pid] = chosen
    return paired


def save(fig, outdir, name, show):
    fig.tight_layout()
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  wrote {path}")
    if not show:
        import matplotlib.pyplot as plt
        plt.close(fig)


def write_csv(outdir, name, header, records):
    path = os.path.join(outdir, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(records)
    print(f"  wrote {path}")


def _empty_axis(ax, msg="no data"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
            color="gray", fontsize=11)


# --------------------------------------------------------------------------- #
# scatter + median-trend panel (shared by specs & convergence overviews)
# --------------------------------------------------------------------------- #
def _scatter_panel(ax, by_method, x, y, *, logy=False, xlabel=None, ylabel=None,
                   title=None):
    drew = False
    for m in METHOD_ORDER:
        rows = [r for r in by_method.get(m, []) if _ok(r, x, y)]
        if not rows:
            continue
        drew = True
        st = METHOD_STYLE[m]
        xv = [r[x] for r in rows]
        yv = [r[y] for r in rows]
        ax.scatter(xv, yv, s=22, alpha=0.35, color=st["color"],
                   marker=st["marker"], label=st["label"], edgecolors="none")
        cx, cy = median_curve(rows, x, y)
        if cx:
            ax.plot(cx, cy, color=st["color"], lw=2.0, marker=st["marker"],
                    markersize=4)
    if not drew:
        _empty_axis(ax)
        return
    if logy:
        ax.set_yscale("log")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, framealpha=0.9)


def _ratio_hist(ax, ratios, *, label, color, unit="x"):
    """Log-x histogram of reduction factors with the median annotated."""
    ratios = [r for r in ratios if r is not None and np.isfinite(r) and r > 0]
    if not ratios:
        _empty_axis(ax)
        ax.set_title(label)
        return
    bins = np.logspace(np.log10(min(ratios)), np.log10(max(ratios)), 24) \
        if max(ratios) > min(ratios) else 12
    ax.hist(ratios, bins=bins, color=color, alpha=0.8, edgecolor="white")
    if isinstance(bins, np.ndarray):
        ax.set_xscale("log")
    med = float(np.median(ratios))
    ax.axvline(med, color="black", ls="--", lw=1.5)
    ax.annotate(f"median ≈ {med:.0f}{unit}", xy=(med, 0.92),
                xycoords=("data", "axes fraction"), ha="left", va="top",
                fontsize=9, fontweight="bold",
                xytext=(4, 0), textcoords="offset points")
    ax.set_title(label)
    ax.set_xlabel(f"reduction factor (integer-HUBO / cardinality) [{unit}]")
    ax.set_ylabel("problems")
    ax.grid(True, ls=":", alpha=0.4)


# --------------------------------------------------------------------------- #
# specs
# --------------------------------------------------------------------------- #
def visualize_specs(rows, outdir, show):
    import matplotlib.pyplot as plt

    usable = [r for r in rows if "error" not in r]
    dropped = len(rows) - len(usable)
    if dropped:
        print(f"  note: skipped {dropped} errored spec row(s)")
    by_method = group_by_method(usable)
    print(f"  rows: {', '.join(f'{m}={len(by_method[m])}' for m in METHOD_ORDER)}")

    # --- Figure 1: resource scaling -------------------------------------- #
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    _scatter_panel(axes[0, 0], by_method, "n_qubits", "depth", logy=True,
                   xlabel="qubits", ylabel="circuit depth",
                   title="Circuit depth vs qubits")
    _scatter_panel(axes[0, 1], by_method, "n_qubits", "two_qubit_gate_count",
                   logy=True, xlabel="qubits", ylabel="2-qubit (CNOT) gates",
                   title="Entangling-gate count vs qubits")
    _scatter_panel(axes[1, 0], by_method, "n_qubits", "total_gates", logy=True,
                   xlabel="qubits", ylabel="total gates",
                   title="Total gate count vs qubits")
    # qubit overhead: n_qubits vs n_assets (+ y=x reference)
    ax = axes[1, 1]
    _scatter_panel(ax, by_method, "n_assets", "n_qubits",
                   xlabel="assets (N)", ylabel="qubits",
                   title="Qubit footprint vs problem size")
    na = [r["n_assets"] for r in usable if _ok(r, "n_assets")]
    if na:
        lim = [min(na), max(na)]
        ax.plot(lim, lim, color="gray", ls="--", lw=1, label="y = x (1 qubit/asset)")
        ax.legend(fontsize=8)
    fig.suptitle("Static circuit-resource scaling — log-encoded HOPO vs ring-XY",
                 fontsize=13, fontweight="bold")
    save(fig, outdir, "specs_scaling.png", show)

    # --- Figure 2: per-problem savings ----------------------------------- #
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    depth_pair = paired_by_problem(usable, "depth")
    cnot_pair = paired_by_problem(usable, "two_qubit_gate_count")
    qubit_pair = paired_by_problem(usable, "n_qubits")

    def _ratios(pair, field, denom_field=None):
        out = []
        for pr in pair.values():
            ih, cd = pr["integer_hubo"], pr["cardinality"]
            num = ih.get(field)
            den = cd.get(denom_field or field)
            if num and den:
                out.append(num / den)
        return out

    _ratio_hist(axes[0], _ratios(depth_pair, "depth"),
                label="Depth reduction", color=METHOD_STYLE["cardinality"]["color"])
    _ratio_hist(axes[1], _ratios(cnot_pair, "two_qubit_gate_count"),
                label="CNOT reduction", color=METHOD_STYLE["cardinality"]["color"])
    # qubit reduction = integer_hubo n_qubits / cardinality n_qubits (== n_assets)
    _ratio_hist(axes[2], _ratios(qubit_pair, "n_qubits"),
                label="Qubit reduction", color=METHOD_STYLE["cardinality"]["color"])
    n_paired = len(depth_pair)
    fig.suptitle(f"Per-problem resource savings (cardinality vs integer-HUBO, "
                 f"{n_paired} paired problems)", fontsize=13, fontweight="bold")
    save(fig, outdir, "specs_savings.png", show)

    # --- Figure 3: gate composition -------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 5.5))
    unexpected = set()
    bottoms = {m: 0.0 for m in METHOD_ORDER}
    xpos = {m: i for i, m in enumerate(METHOD_ORDER)}
    mean_frac = {}  # method -> {gate: mean fraction}
    for m in METHOD_ORDER:
        fracs = defaultdict(list)
        for r in by_method[m]:
            gc = r.get("gate_counts") or {}
            tot = sum(gc.values())
            for u in (r.get("unexpected_gates") or []):
                unexpected.add(u)
            if tot <= 0:
                continue
            for g in GATE_TYPES:
                fracs[g].append(gc.get(g, 0) / tot)
        mean_frac[m] = {g: (float(np.mean(fracs[g])) if fracs[g] else 0.0)
                        for g in GATE_TYPES}
    for g in GATE_TYPES:
        vals = [mean_frac[m][g] for m in METHOD_ORDER]
        if not any(vals):
            continue
        ax.bar([xpos[m] for m in METHOD_ORDER], vals,
               bottom=[bottoms[m] for m in METHOD_ORDER],
               color=GATE_COLORS[g], label=g, width=0.55, edgecolor="white")
        for m in METHOD_ORDER:
            bottoms[m] += mean_frac[m][g]
    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels([METHOD_STYLE[m]["label"] for m in METHOD_ORDER])
    ax.set_ylabel("mean gate-type fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Gate composition by method")
    ax.legend(title="gate", fontsize=9)
    if unexpected:
        ax.text(0.0, -0.12, f"unexpected gates present: {sorted(unexpected)}",
                transform=ax.transAxes, fontsize=8, color="gray")
    save(fig, outdir, "specs_gate_composition.png", show)

    # --- CSV summary ----------------------------------------------------- #
    records = []
    for m in METHOD_ORDER:
        buckets = defaultdict(list)
        for r in by_method[m]:
            if _ok(r, "n_qubits"):
                buckets[r["n_qubits"]].append(r)
        for nq in sorted(buckets):
            grp = buckets[nq]
            def med(field):
                xs = [g[field] for g in grp if _ok(g, field)]
                return round(float(np.median(xs)), 2) if xs else ""
            records.append([m, nq, len(grp), med("depth"),
                            med("two_qubit_gate_count"), med("total_gates"),
                            med("num_params")])
    write_csv(outdir, "specs_summary.csv",
              ["method", "n_qubits", "count", "median_depth", "median_cnots",
               "median_total_gates", "median_num_params"], records)


# --------------------------------------------------------------------------- #
# convergence
# --------------------------------------------------------------------------- #
def quality(row):
    """Unified solution quality in [0, 1], higher = better (1 = optimal).

    Both methods store the *global* min-max ratio (final - E_min)/(E_max - E_min),
    where 0 = optimal: ``approximation_ratio`` (integer_hubo, profile.py) and
    ``approximation_ratio_global`` (cardinality, ring_xy.py). This returns
    ``1 - ratio`` so figures/CSV read higher-is-better. Do NOT switch the
    cardinality branch to ``approximation_ratio_subspace`` — it has the opposite
    orientation (1 = optimal vs a Dicke-mean baseline) and is not comparable
    across methods. None if unusable or infeasible.
    """
    if row.get("infeasible"):
        return None
    if row.get("method") == "cardinality":
        r = row.get("approximation_ratio_global")
    else:
        r = row.get("approximation_ratio")
    # same corrupt-ratio guard as profile.py, for results files predating it
    if r is None or not (-0.01 <= r <= 1.01):
        return None
    return 1.0 - r


# --------------------------------------------------------------------------- #
# Fig. 6 comparison: normalized portfolio objective vs budget utilization
# --------------------------------------------------------------------------- #
def _normalize_compare(rows):
    """Per-problem min-max normalization of the portfolio objective f(z).

    f_min/f_max are taken over the candidate set {budget-feasible method
    post_objectives, continuous, continuous_unconstrained, reference}. The exact
    reference is treated as a *plain series*, NOT asserted optimal (an
    over-budget integer-HUBO solution searches integer vectors directly and can
    exceed the allocator-family reference). Mutates each usable method row with
    ``_norm_objective`` (computed for over-budget rows too, against the feasible
    bounds, so they can be plotted at y>1); returns (norm_by_pid, n_beat_reference).

    Normalization is for *visualization only* — claims rest on the
    normalization-invariant sign of (f_method - f_continuous). A relative-to-scale
    degeneracy guard skips problems whose candidate spread is numerically zero.
    """
    by_pid = defaultdict(list)
    for r in rows:
        pid = r.get("problem_id")
        if pid is not None and r.get("method") in METHOD_ORDER and "error" not in r:
            by_pid[pid].append(r)

    norm_by_pid = {}
    n_beat_reference = 0
    for pid, group in by_pid.items():
        base = group[0]  # enrichment is identical across a problem's rows
        cont_obj = base.get("continuous_objective")
        cont_unc_obj = base.get("continuous_unconstrained_objective")
        ref_obj = base.get("reference_objective")

        # Collapse multi-K (e.g. --sweep) rows to one representative per method so
        # extra K rows can't pollute the f_min/f_max normalization bounds (mirrors
        # paired_by_problem). Continuous/unconstrained/reference are per-problem
        # scalars already; only the method post_objectives need collapsing.
        feasible_by_method = defaultdict(list)
        for r in group:
            po = r.get("post_objective")
            if _finite(po) and not r.get("over_budget"):
                feasible_by_method[r["method"]].append(po)
        cands = []
        for vals in feasible_by_method.values():
            target = float(np.median(vals))
            cands.append(min(vals, key=lambda v: abs(v - target)))
        cands += [v for v in (cont_obj, cont_unc_obj, ref_obj) if _finite(v)]

        info = {
            "ok": False, "f_min": None, "f_max": None,
            "total_budget": base.get("total_budget"),
            "continuous_objective": cont_obj,
            "continuous_budget_util": base.get("continuous_budget_util"),
            "continuous_norm": None,
            "continuous_unconstrained_objective": cont_unc_obj,
            "continuous_unconstrained_budget_util":
                base.get("continuous_unconstrained_budget_util"),
            "continuous_unconstrained_norm": None,
            "reference_objective": ref_obj,
            "reference_budget_util": base.get("reference_budget_util"),
            "reference_norm": None,
            "reason": None,  # why a problem was dropped (info["ok"] stays False)
        }

        # C2: surface (don't hide) any method beating the exact reference.
        if _finite(ref_obj):
            for r in group:
                po = r.get("post_objective")
                if _finite(po) and po > ref_obj:
                    n_beat_reference += 1
                    print(f"  note: problem {pid} {r.get('method')} f(z)={po:.6g} "
                          f"exceeds exact reference {ref_obj:.6g} "
                          f"(over_budget={bool(r.get('over_budget'))})")

        if len(cands) >= 2:
            f_min, f_max = min(cands), max(cands)
            # I3: relative-to-scale degeneracy guard (not mere distinctness).
            if (f_max - f_min) > 1e-9 * max(abs(f_max), abs(f_min), 1.0):
                span = f_max - f_min

                def _norm(x):
                    return (x - f_min) / span if _finite(x) else None

                info.update(ok=True, f_min=f_min, f_max=f_max)
                info["continuous_norm"] = _norm(cont_obj)
                info["continuous_unconstrained_norm"] = _norm(cont_unc_obj)
                info["reference_norm"] = _norm(ref_obj)
                for r in group:
                    r["_norm_objective"] = _norm(r.get("post_objective"))
            else:
                info["reason"] = "degenerate objective spread (~0)"
        else:
            info["reason"] = "fewer than 2 comparable objectives"
        norm_by_pid[pid] = info
    return norm_by_pid, n_beat_reference


def _is_feasible(row):
    """A method row is budget-feasible when its utilization is known and <= 1+tol.
    Continuous/ring-XY are LP-capped; only integer-HUBO can overspend."""
    bu = row.get("budget_util")
    return _finite(bu) and bu <= 1.0 + COMPARE_BUDGET_TOL


def _compare_points(rows, norm_by_pid):
    """Collect (budget_util, norm_objective) per series for compare_fig6."""
    series = {k: {"x": [], "y": []} for k in COMPARE_STYLE}
    for r in rows:
        if r.get("method") not in METHOD_ORDER or "error" in r:
            continue
        if not _ok(r, "_norm_objective", "budget_util"):
            continue
        key = "over_budget" if r.get("over_budget") else r["method"]
        series[key]["x"].append(r["budget_util"])
        series[key]["y"].append(r["_norm_objective"])
    for info in norm_by_pid.values():
        if not info.get("ok"):
            continue
        for key, ub, nb in (
            ("continuous", "continuous_budget_util", "continuous_norm"),
            ("continuous_unconstrained", "continuous_unconstrained_budget_util",
             "continuous_unconstrained_norm"),
            ("reference", "reference_budget_util", "reference_norm")):
            x, y = info.get(ub), info.get(nb)
            if _finite(x) and _finite(y):
                series[key]["x"].append(x)
                series[key]["y"].append(y)
    return series


def _draw_compare(usable, outdir, show):
    """Fig. 6 reproduction + head-to-head face-off + CSV. Gated by the caller on
    the presence of the continuous-baseline enrichment (older files lack it)."""
    import matplotlib.pyplot as plt

    norm_by_pid, n_beat_ref = _normalize_compare(usable)
    n_ok = sum(1 for i in norm_by_pid.values() if i["ok"])
    dropped = sorted((pid, i.get("reason") or "not normalizable")
                     for pid, i in norm_by_pid.items() if not i["ok"])
    n_over = sum(1 for r in usable
                 if r.get("method") in METHOD_ORDER and r.get("over_budget"))
    print(f"  compare: {n_ok} problem(s) normalized; {n_over} over-budget point(s); "
          f"{n_beat_ref} reference-beating point(s)")
    if dropped:
        print("  note: omitted from compare_fig6 (not normalizable): "
              + ", ".join(f"#{pid} ({why})" for pid, why in dropped))
    if n_ok == 0:
        print("  note: no problem had >=2 comparable objectives; "
              "skipping compare figures")
        return

    # --- compare_fig6: normalized objective vs budget utilization ------------ #
    fig, ax = plt.subplots(figsize=(8.5, 6))
    series = _compare_points(usable, norm_by_pid)
    plot_order = ["continuous", "continuous_unconstrained", "reference",
                  "integer_hubo", "cardinality", "over_budget"]
    drew = False
    for key in plot_order:
        pts = series[key]
        if not pts["x"]:
            continue
        drew = True
        st = COMPARE_STYLE[key]
        ax.scatter(pts["x"], pts["y"], s=38, alpha=0.6, color=st["color"],
                   marker=st["marker"],
                   label=f"{st['label']} (n={len(pts['x'])})", edgecolors="none")
    if drew:
        ax.axvline(1.0, color="gray", ls="--", lw=1, label="budget limit (util = 1)")
        ax.set_xlabel("budget utilization (realized / budget)")
        ax.set_ylabel("min-max normalized objective f(z)  (per problem)")
        ax.set_title("Portfolio objective vs budget utilization (paper Fig. 6)")
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(fontsize=8, framealpha=0.9, loc="best")
    else:
        _empty_axis(ax)
    if dropped:
        omitted = ", ".join(f"#{pid} ({why})" for pid, why in dropped)
        fig.text(0.5, 0.01, f"omitted (not normalizable): {omitted}",
                 ha="center", va="bottom", fontsize=7, color="gray", wrap=True)
        fig.subplots_adjust(bottom=0.13)
    save(fig, outdir, "compare_fig6.png", show)

    # --- compare_faceoff: per method vs continuous (feasible points only) ---- #
    fig, axes = plt.subplots(1, len(METHOD_ORDER), figsize=(11, 5))
    win_stats = {}  # method -> {pid: bool} over feasible, continuous-paired points
    for ax, m in zip(np.atleast_1d(axes), METHOD_ORDER):
        xs, ys, pids = [], [], []
        for r in usable:
            if r.get("method") != m or "error" in r or not _is_feasible(r):
                continue
            if not _ok(r, "_norm_objective"):
                continue
            info = norm_by_pid.get(r.get("problem_id"))
            if info is None or not _finite(info.get("continuous_norm")):
                continue
            xs.append(info["continuous_norm"])
            ys.append(r["_norm_objective"])
            pids.append(r.get("problem_id"))
        # sign is normalization-invariant: y>x iff f_method > f_continuous
        wins = {p: (y > x) for p, x, y in zip(pids, xs, ys)}
        win_stats[m] = wins
        st = METHOD_STYLE[m]
        if xs:
            ax.scatter(xs, ys, s=34, alpha=0.65, color=st["color"],
                       marker=st["marker"], edgecolors="none")
            lim = [min(xs + ys), max(xs + ys)]
            ax.plot(lim, lim, color="gray", ls="--", lw=1, label="y = x (tie)")
            nwin = sum(wins.values())
            ax.set_title(f"{st['label']}\nwin vs continuous: {nwin}/{len(xs)} "
                         f"= {nwin / len(xs):.0%} (feasible)")
            ax.set_xlabel("continuous normalized objective")
            ax.set_ylabel(f"{st['label']} normalized objective")
            ax.legend(fontsize=8)
            ax.grid(True, ls=":", alpha=0.4)
        else:
            _empty_axis(ax)
            ax.set_title(f"{st['label']} — no feasible paired points")
    fig.suptitle("Head-to-head vs continuous baseline (above line beats continuous)",
                 fontsize=13, fontweight="bold")
    save(fig, outdir, "compare_faceoff.png", show)

    # --- per-method win-rate (feasible) + common-set comparison (I2) --------- #
    print("  win-rate vs continuous (budget-feasible points):")
    for m in METHOD_ORDER:
        wins = win_stats.get(m, {})
        n = len(wins)
        if n:
            print(f"    {m}: {sum(wins.values())}/{n} = {sum(wins.values()) / n:.0%}")
        else:
            print(f"    {m}: no feasible paired points")
    common = set(win_stats.get(METHOD_ORDER[0], {}))
    for m in METHOD_ORDER[1:]:
        common &= set(win_stats.get(m, {}))
    if common:
        print(f"  common feasible problem set (n={len(common)}):")
        for m in METHOD_ORDER:
            w = sum(win_stats[m][p] for p in common)
            print(f"    {m}: {w}/{len(common)} = {w / len(common):.0%}")
    else:
        print("  common feasible problem set: empty (no cross-method comparison)")

    # --- compare_summary.csv ------------------------------------------------- #
    records = []
    for r in usable:
        if r.get("method") not in METHOD_ORDER or "error" in r:
            continue
        info = norm_by_pid.get(r.get("problem_id")) or {}
        po = r.get("post_objective")
        cont_obj = info.get("continuous_objective")
        beats = (po > cont_obj) if (_finite(po) and _finite(cont_obj)) else ""
        cn = info.get("continuous_norm")
        no = r.get("_norm_objective")
        delta = (no - cn) if (_finite(no) and _finite(cn)) else ""

        def _r(x):
            return round(float(x), 6) if _finite(x) else ""

        records.append([
            r.get("problem_id"), r.get("n_assets"), _r(info.get("total_budget")),
            r.get("method"), _r(po), _r(r.get("budget_util")),
            _is_feasible(r), bool(r.get("over_budget")),
            _r(no), _r(cont_obj), _r(cn), beats, _r(delta),
        ])
    write_csv(outdir, "compare_summary.csv",
              ["problem_id", "n_assets", "total_budget", "method", "post_objective",
               "budget_util", "budget_feasible", "over_budget", "norm_objective",
               "continuous_objective", "continuous_norm", "beats_continuous",
               "delta_vs_continuous_norm_dependent"], records)

    # Benefit-over-continuous as a function of problem size N (the scaling story the
    # Fig. 6 budget-utilization view can't show). Rides this function's
    # continuous-baseline gate; uses raw post_objective / returns (NOT the endogenous
    # Fig. 6 span), so it is unaffected by _normalize_compare's degeneracy drops.
    _draw_benefit_by_n(usable, outdir, show)


# --------------------------------------------------------------------------- #
# benefit vs asset count N: win-rate + relative utility gap + price-weighted
# profit gap, each vs the classical continuous baseline. Magnitudes use
# problem-EXTERNAL denominators (|continuous| objective; total budget) so the
# N-trend is interpretable — unlike the Fig. 6 normalized Δ, whose per-problem
# min-max span is endogenous and scales with N.
# --------------------------------------------------------------------------- #
BENEFIT_MIN_N = 3        # min problems in a (method, N) cell to draw a median+IQR band
BENEFIT_REL_TOL = 1e-9   # relative guard for the |continuous_objective| denominator


def _iqr(vals):
    q1, q3 = np.percentile(vals, [25, 75])
    return float(q3 - q1)


def _set_robust_ylim(ax, vals):
    """Clip a gap axis to its central 5–95th percentile (always keeping y=0 in view)
    so a few extreme outliers — e.g. the relative utility gap exploding when
    |continuous| is small-but-finite — don't crush the readable trend. Medians/IQR
    are still computed on the full data; off-scale points are counted, not dropped."""
    vals = [v for v in vals if _finite(v)]
    if len(vals) < 2:
        return
    lo, hi = (float(x) for x in np.percentile(vals, [5, 95]))
    if hi <= lo:
        return
    pad = 0.1 * (hi - lo)
    lo, hi = min(lo - pad, 0.0), max(hi + pad, 0.0)
    n_off = sum(1 for v in vals if v < lo or v > hi)
    ax.set_ylim(lo, hi)
    if n_off:
        ax.text(0.99, 0.02, f"{n_off} pt(s) off-scale", transform=ax.transAxes,
                fontsize=7, color="gray", ha="right", va="bottom")


def _benefit_records(usable):
    """Per-(method, problem) benefit records vs the continuous baseline. Feasible
    points only; one record per problem (multi-K collapsed by the median
    post_objective, mirroring paired_by_problem) so a swept problem can't dominate
    an N bucket. Returns {method: [{N, win, util_gap, profit_gap}, ...]}.

    - win: post_method > continuous_objective (sign only, rigorous).
    - util_gap: (post_method - continuous)/|continuous|, with a RELATIVE denominator
      guard (skip near-zero |continuous|); external denominator -> N-trend valid.
    - profit_gap: (method_return - continuous_return)/total_budget, both price-weighted
      expected returns in $/yr -> a dimensionless annual rate. None when the profit
      enrichment is absent (older JSON / offline / integer-HUBO with no allocation).
    """
    by_pm = defaultdict(lambda: defaultdict(list))
    for r in usable:
        m = r.get("method")
        if m not in METHOD_ORDER or "error" in r or not _is_feasible(r):
            continue
        if not _ok(r, "post_objective", "n_assets"):
            continue
        by_pm[m][r.get("problem_id")].append(r)

    out = {m: [] for m in METHOD_ORDER}
    for m in METHOD_ORDER:
        for cands in by_pm[m].values():
            vals = [c["post_objective"] for c in cands]
            target = float(np.median(vals))
            row = min(cands, key=lambda c: abs(c["post_objective"] - target))
            po, cont = row["post_objective"], row.get("continuous_objective")
            rec = {"N": row["n_assets"], "win": None,
                   "util_gap": None, "profit_gap": None}
            if _finite(po) and _finite(cont):
                rec["win"] = bool(po > cont)
                if abs(cont) > BENEFIT_REL_TOL * max(abs(po), abs(cont), 1.0):
                    rec["util_gap"] = (po - cont) / abs(cont)
            mret, cret, tb = (row.get("method_return"),
                              row.get("continuous_return"), row.get("total_budget"))
            if _finite(mret) and _finite(cret) and _finite(tb) and tb > 0:
                rec["profit_gap"] = (mret - cret) / tb
            out[m].append(rec)
    return out


def _draw_benefit_by_n(usable, outdir, show):
    import matplotlib.pyplot as plt

    recs = _benefit_records(usable)
    n_records = sum(len(v) for v in recs.values())
    has_profit = any(r["profit_gap"] is not None for v in recs.values() for r in v)
    print(f"  benefit-vs-N: {n_records} feasible problem-record(s); "
          f"profit data {'present' if has_profit else 'absent'}")
    if n_records == 0:
        print("  note: no feasible benefit records; skipping benefit_by_n")
        return

    fig, (ax_win, ax_util, ax_profit) = plt.subplots(1, 3, figsize=(16, 5))

    def _bucket(method, field):
        b = defaultdict(list)
        for r in recs[method]:
            v = r[field]
            if v is None:
                continue
            if field != "win" and not _finite(v):
                continue
            b[r["N"]].append(v)
        return b

    def _plot_winrate(ax, b, st):
        if not b:
            return
        xs = sorted(b)
        for n in xs:
            ax.annotate(f"n={len(b[n])}", (n, float(np.mean(b[n]))), fontsize=7,
                        color=st["color"], xytext=(0, 5),
                        textcoords="offset points", ha="center")
        ax.plot(xs, [float(np.mean(b[n])) for n in xs], color=st["color"],
                marker=st["marker"], lw=2.0)

    def _plot_gap(ax, b, st):
        """Scatter every per-problem point (+ n annotation). A median trend line and
        IQR band are drawn only across cells with n >= BENEFIT_MIN_N; sparse cells
        (n < MIN_N) are left as bare points so a 1-2 sample band can't read as fact."""
        if not b:
            return
        for n in sorted(b):
            pts = b[n]
            ax.scatter([n] * len(pts), pts, s=16, alpha=0.30, color=st["color"],
                       marker=st["marker"], edgecolors="none")
            ax.annotate(f"n={len(pts)}", (n, float(np.median(pts))), fontsize=7,
                        color=st["color"], xytext=(0, 5),
                        textcoords="offset points", ha="center")
        robust = [n for n in sorted(b) if len(b[n]) >= BENEFIT_MIN_N]
        if robust:
            ax.plot(robust, [float(np.median(b[n])) for n in robust],
                    color=st["color"], marker=st["marker"], lw=2.0)
            ax.fill_between(robust,
                            [float(np.percentile(b[n], 25)) for n in robust],
                            [float(np.percentile(b[n], 75)) for n in robust],
                            color=st["color"], alpha=0.15)

    csv_rows = []
    util_all, profit_all = [], []
    for m in METHOD_ORDER:
        st = METHOD_STYLE[m]
        win_b, util_b, profit_b = (_bucket(m, "win"), _bucket(m, "util_gap"),
                                   _bucket(m, "profit_gap"))
        util_all += [v for vs in util_b.values() for v in vs]
        profit_all += [v for vs in profit_b.values() for v in vs]
        # one labeled proxy per method per axis so every method shows in the legend
        for ax in (ax_win, ax_util, ax_profit):
            ax.plot([], [], color=st["color"], marker=st["marker"], lw=2.0,
                    label=st["label"])
        _plot_winrate(ax_win, win_b, st)
        _plot_gap(ax_util, util_b, st)
        _plot_gap(ax_profit, profit_b, st)
        for n in sorted(set(win_b) | set(util_b) | set(profit_b)):
            w, u, p = win_b.get(n, []), util_b.get(n, []), profit_b.get(n, [])
            csv_rows.append([
                m, n,
                len(w), int(sum(w)),
                round(float(np.mean(w)), 6) if w else "",
                len(u), round(float(np.median(u)), 6) if u else "",
                round(_iqr(u), 6) if u else "",
                len(p), round(float(np.median(p)), 6) if p else "",
                round(_iqr(p), 6) if p else "",
            ])

    ax_win.axhline(0.5, color="gray", ls="--", lw=1, label="50% (coin flip)")
    ax_win.set_ylim(0, 1)
    ax_win.set_ylabel("win rate vs continuous (fraction post > continuous, feasible)")
    ax_win.set_title("Win rate vs N")

    ax_util.axhline(0.0, color="gray", ls="--", lw=1, label="tie (Δ=0)")
    ax_util.set_ylabel("relative utility gap  (post − cont) / |cont|")
    ax_util.set_title("Utility benefit vs N")
    # The relative utility gap is heavy-tailed: when |continuous| is small the ratio
    # blows up, so a linear axis is dominated by a few extreme integer-HUBO points and
    # the bulk (and both median trends) collapse onto y=0. A symlog axis keeps the
    # near-zero region — where almost every point lives — linear and readable while
    # still showing the tail (no clipping, no information lost). linthresh adapts to
    # the typical gap magnitude so the linear band frames the real signal.
    _nz = [abs(v) for v in util_all if _finite(v) and abs(v) > 0]
    linthresh = min(max(float(np.median(_nz)) if _nz else 1.0, 1e-3), 10.0)
    ax_util.set_yscale("symlog", linthresh=linthresh)

    if has_profit:
        ax_profit.axhline(0.0, color="gray", ls="--", lw=1, label="tie (Δ=0)")
        ax_profit.set_ylabel("profit gap  Δ(expected return) / budget  [annual rate]")
        ax_profit.set_title("Profit benefit vs N")
        _set_robust_ylim(ax_profit, profit_all)
    else:
        _empty_axis(ax_profit,
                    "profit data unavailable\n(run with live μ / results JSON)")
        ax_profit.set_title("Profit benefit vs N")

    for ax in (ax_win, ax_util, ax_profit):
        ax.set_xlabel("assets (N)")
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(fontsize=8, framealpha=0.9)

    fig.suptitle("Portfolio benefit over continuous baseline vs asset count N "
                 "(feasible points)", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.005,
             "Feasible points only. Utility gap uses the external |continuous| "
             "denominator (sign + magnitude both interpretable). Profit gap is the "
             "price-weighted EXPECTED annual return advantage ÷ total budget — a "
             "joint deployment + selection measure (idle cash scores 0), not pure "
             "stock-picking skill. Median+IQR bands drawn only for n ≥ "
             f"{BENEFIT_MIN_N}; sparser cells shown as bare points (see n labels).",
             ha="center", va="bottom", fontsize=7, color="gray", wrap=True)
    fig.subplots_adjust(bottom=0.16)
    save(fig, outdir, "benefit_by_n.png", show)

    write_csv(outdir, "benefit_by_n.csv",
              ["method", "n_assets", "n_winrate", "wins", "win_rate",
               "n_util", "median_util_gap", "iqr_util_gap",
               "n_profit", "median_profit_gap", "iqr_profit_gap"], csv_rows)


def visualize_convergence(rows, outdir, show):
    import matplotlib.pyplot as plt

    usable = [r for r in rows if "error" not in r]
    dropped = len(rows) - len(usable)
    if dropped:
        print(f"  note: skipped {dropped} errored convergence row(s)")
    by_method = group_by_method(usable)
    infeasible = sum(1 for r in usable if r.get("infeasible"))
    print(f"  rows: {', '.join(f'{m}={len(by_method[m])}' for m in METHOD_ORDER)}"
          f" (infeasible: {infeasible})")

    # attach a unified 'quality' field so the shared panel can read it
    for r in usable:
        r["_quality"] = quality(r)

    # --- Figure 1: overview ---------------------------------------------- #
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    # Quality is compared on the method-independent problem-size axis (n_assets),
    # NOT n_qubits: cardinality uses n_qubits == N while integer-HUBO log-encodes
    # to n_qubits > N, so the same problem lands at different x for the two
    # methods and equal-qubit points are different-sized problems. The remaining
    # panels stay on n_qubits — effort scales with the circuit, not N.
    _scatter_panel(axes[0], by_method, "n_assets", "_quality",
                   xlabel="assets (N)",
                   ylabel="solution quality = 1 − min-max ratio (1 = optimal)",
                   title="Solution quality vs problem size")
    _scatter_panel(axes[1], by_method, "n_qubits", "evaluations",
                   xlabel="qubits", ylabel="CMA-ES evaluations",
                   title="Optimizer effort vs qubits")
    _scatter_panel(axes[2], by_method, "n_qubits", "post_objective",
                   xlabel="qubits", ylabel="post-allocation objective",
                   title="Objective vs qubits")
    suffix = f" — {infeasible} infeasible row(s) omitted" if infeasible else ""
    fig.suptitle(f"QAOA convergence summary — HOPO vs ring-XY{suffix}",
                 fontsize=13, fontweight="bold")
    save(fig, outdir, "convergence_overview.png", show)

    # --- Figure 2: per-problem savings + quality face-off ---------------- #
    wall_pair = paired_by_problem(usable, "wall_clock_seconds")
    speedups = []
    for pr in wall_pair.values():
        ih = pr["integer_hubo"].get("wall_clock_seconds")
        cd = pr["cardinality"].get("wall_clock_seconds")
        if ih and cd:
            speedups.append(ih / cd)

    # Drop the wall-clock panel entirely when no row carries timing (e.g. both
    # --hopo-source json and --card-source json, which have no per-K timing) so
    # the figure shows only the quality face-off instead of an empty "no data" box.
    has_wall = bool(speedups)
    if has_wall:
        fig, (wall_ax, ax) = plt.subplots(1, 2, figsize=(11, 4.8))
        _ratio_hist(wall_ax, speedups, label="Wall-clock speedup",
                    color=METHOD_STYLE["cardinality"]["color"], unit="x")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5.8, 4.8))

    # quality face-off: cardinality (y) vs integer_hubo (x), y=x ref
    q_pair = paired_by_problem(usable, "_quality")
    xs, ys = [], []
    for pr in q_pair.values():
        x = pr["integer_hubo"].get("_quality")
        y = pr["cardinality"].get("_quality")
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    if xs:
        ax.scatter(xs, ys, s=30, alpha=0.6,
                   color=METHOD_STYLE["cardinality"]["color"], edgecolors="none")
        lim = [min(xs + ys), max(xs + ys)]
        ax.plot(lim, lim, color="gray", ls="--", lw=1, label="y = x (tie)")
        ax.set_xlabel("integer-HUBO quality (1 = optimal)")
        ax.set_ylabel("cardinality quality (1 = optimal)")
        ax.set_title("Quality face-off (above line: cardinality wins)")
        ax.legend(fontsize=8)
        ax.grid(True, ls=":", alpha=0.4)
    else:
        _empty_axis(ax)
        ax.set_title("Quality face-off")
    n_paired = len(wall_pair) if has_wall else len(q_pair)
    fig.suptitle(f"Per-problem convergence comparison ({n_paired} paired)",
                 fontsize=13, fontweight="bold")
    save(fig, outdir, "convergence_savings.png", show)

    # --- CSV summary ----------------------------------------------------- #
    records = []
    for m in METHOD_ORDER:
        buckets = defaultdict(list)
        for r in by_method[m]:
            if _ok(r, "n_qubits"):
                buckets[r["n_qubits"]].append(r)
        for nq in sorted(buckets):
            grp = buckets[nq]
            def med(field):
                xs = [g[field] for g in grp if _ok(g, field)]
                return round(float(np.median(xs)), 4) if xs else ""
            qs = [g["_quality"] for g in grp if g.get("_quality") is not None]
            records.append([
                m, nq, len(grp),
                round(float(np.median(qs)), 4) if qs else "",
                med("wall_clock_seconds"), med("evaluations"),
                sum(1 for g in grp if g.get("infeasible")),
            ])
    write_csv(outdir, "convergence_summary.csv",
              ["method", "n_qubits", "count", "median_quality",
               "median_wall_clock_s", "median_evaluations", "infeasible"], records)

    # --- Fig. 6 comparison (vs continuous baseline) ---------------------- #
    # Gated on the additive enrichment: older convergence JSONs (pre-continuous)
    # simply skip it, so existing figures/CSV above are unaffected.
    if any(r.get("continuous_objective") is not None for r in usable):
        _draw_compare(usable, outdir, show)
    else:
        print("  note: no continuous-baseline fields in rows "
              "(older profile JSON?); skipping Fig. 6 comparison")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def _load_and_visualize(path, expected_sub, payload_key, visualize, outdir, show):
    """Load one profile JSON, check it matches ``expected_sub``, and draw it."""
    if not os.path.exists(path):
        sys.exit(f"error: {path} not found")
    with open(path) as f:
        doc = json.load(f)

    meta = doc.get("meta", {})
    sub = meta.get("subcommand")
    print(f"Using {expected_sub} file: {path} "
          f"(subcommand={sub!r}, method={meta.get('method')!r})")

    if sub != expected_sub or payload_key not in doc:
        sys.exit(f"error: {path} is not a {expected_sub!r} profile. "
                 f"meta.subcommand={sub!r}; top-level keys={list(doc.keys())}")
    visualize(doc[payload_key], outdir, show)


def main():
    ap = argparse.ArgumentParser(description="Visualize profile_results.json")
    ap.add_argument("--specs", help="path to a `profile.py specs` JSON file")
    ap.add_argument("--convergence",
                    help="path to a `profile.py convergence` JSON file")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--show", action="store_true",
                    help="display figures interactively instead of headless save")
    args = ap.parse_args()

    if not args.specs and not args.convergence:
        sys.exit("error: specify at least one of --specs PATH or --convergence PATH")

    if not args.show:
        matplotlib.use("Agg")

    os.makedirs(args.outdir, exist_ok=True)

    if args.specs:
        _load_and_visualize(args.specs, "specs", "static_specs",
                            visualize_specs, args.outdir, args.show)
    if args.convergence:
        _load_and_visualize(args.convergence, "convergence", "convergence",
                            visualize_convergence, args.outdir, args.show)

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
