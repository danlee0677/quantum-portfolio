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
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    # Quality is compared on the method-independent problem-size axis (n_assets),
    # NOT n_qubits: cardinality uses n_qubits == N while integer-HUBO log-encodes
    # to n_qubits > N, so the same problem lands at different x for the two
    # methods and equal-qubit points are different-sized problems. The remaining
    # panels stay on n_qubits — runtime/effort scale with the circuit, not N.
    _scatter_panel(axes[0, 0], by_method, "n_assets", "_quality",
                   xlabel="assets (N)",
                   ylabel="solution quality = 1 − min-max ratio (1 = optimal)",
                   title="Solution quality vs problem size")
    _scatter_panel(axes[0, 1], by_method, "n_qubits", "wall_clock_seconds",
                   logy=True, xlabel="qubits", ylabel="wall-clock seconds",
                   title="Runtime vs qubits")
    _scatter_panel(axes[1, 0], by_method, "n_qubits", "evaluations",
                   xlabel="qubits", ylabel="CMA-ES evaluations",
                   title="Optimizer effort vs qubits")
    _scatter_panel(axes[1, 1], by_method, "n_qubits", "post_objective",
                   xlabel="qubits", ylabel="post-allocation objective",
                   title="Objective vs qubits")
    suffix = f" — {infeasible} infeasible row(s) omitted" if infeasible else ""
    fig.suptitle(f"QAOA convergence summary — HOPO vs ring-XY{suffix}",
                 fontsize=13, fontweight="bold")
    save(fig, outdir, "convergence_overview.png", show)

    # --- Figure 2: per-problem savings + quality face-off ---------------- #
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    wall_pair = paired_by_problem(usable, "wall_clock_seconds")
    speedups = []
    for pr in wall_pair.values():
        ih = pr["integer_hubo"].get("wall_clock_seconds")
        cd = pr["cardinality"].get("wall_clock_seconds")
        if ih and cd:
            speedups.append(ih / cd)
    _ratio_hist(axes[0], speedups, label="Wall-clock speedup",
                color=METHOD_STYLE["cardinality"]["color"], unit="x")

    # quality face-off: cardinality (y) vs integer_hubo (x), y=x ref
    ax = axes[1]
    q_pair = paired_by_problem(usable, "_quality")
    xs, ys = [], []
    for pr in q_pair.values():
        x = quality(pr["integer_hubo"])
        y = quality(pr["cardinality"])
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
    fig.suptitle(f"Per-problem convergence comparison ({len(wall_pair)} paired)",
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
