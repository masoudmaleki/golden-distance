"""
================================================================================
  GOLDEN DISTANCE (GD) — Python Reference Implementation
  Version 1.0  |  2026
================================================================================

  Paper
  ─────
    Melek M. & Melek N. (2025).
    "Golden Distance: A New and Comprehensive Metric Definition Study
     Facilitating Classification Performance Evaluations."
    Iranian Journal of Science and Technology,
    Transactions of Electrical Engineering.
    https://doi.org/10.1007/s40998-025-00870-x

  Quick start
  ───────────
    from golden_distance import run_golden_distance

    systems = [
        [0.9444, 0.9537, 0.9533, 0.9488],   # Classifier A
        [0.9259, 0.8889, 0.8929, 0.9091],   # Classifier B
    ]
    metric_names = ["ACC", "Sensitivity", "Specificity", "F1"]
    system_names = ["Method A", "Method B"]

    run_golden_distance(systems, metric_names, system_names)

  Citation
  ────────
    If you use this implementation in your research, please cite the paper above.

  Supported range
  ───────────────
    N = 3 … 10 metrics per classifier.
    N ≤ 8  →  exact enumeration of all N! permutations (≤ 40 320).
    N > 8  →  Monte Carlo sampling (default 100 000 draws).

  Mathematical note
  ─────────────────
    The mean NAV has an exact closed form independent of permutation order:
        Av-NAV  =  μ²  −  σ² / (N − 1)
    where μ = mean(metrics) and σ² = variance(metrics).
    Both the numeric permutation average and this closed form are provided
    as a built-in consistency check.

  Requirements
  ────────────
    numpy >= 1.21
    matplotlib >= 3.5
================================================================================
"""

from __future__ import annotations

import os
import warnings
from math import factorial
from itertools import permutations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────
EXACT_LIMIT = 8          # N ≤ this value → full N! enumeration
N_SAMPLES   = 100_000    # Monte Carlo draws when N > EXACT_LIMIT
MAX_DOTS    = 5_000      # max scatter-plot points (subsampled for large N)

# Colour palette (accessible, publication-quality)
_C = dict(
    cloud  = "#AED6F1",   # permutation scatter cloud
    mean   = "#1ABC9C",   # mean (Av-NAV, Av-DCO) point
    ideal  = "#E74C3C",   # ideal point (1, 0)
    line   = "#2C3E50",   # GD dashed connecting line
    poly   = "#2980B9",   # radar polygon fill
    reg    = "#BDC3C7",   # regular (ideal) polygon outline
    grid   = "#D5D8DC",   # concentric circle / spoke colour
    gold   = "#F1C40F",   # 1st-place bar
    silver = "#95A5A6",   # 2nd-place bar
    bronze = "#CA6F1E",   # 3rd-place bar
    bar    = "#3498DB",   # default bar colour
    bg     = "#FDFEFE",   # figure background
    hdr    = "#2C3E50",   # table header background
)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Low-level vectorised mathematics
# ══════════════════════════════════════════════════════════════════════════════

def _vertex_angles(N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Cosine and sine of the N equally-spaced vertex angles of a regular N-gon,
    starting at 0° (positive x-axis) and going counter-clockwise.
    """
    theta = np.arange(N) * (2.0 * np.pi / N)
    return np.cos(theta), np.sin(theta)


def _batch_nav_dco(r: np.ndarray,
                   perm_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised NAV and DCO for every permutation in perm_idx.

    Parameters
    ----------
    r        : (N,)    metric values
    perm_idx : (M, N)  integer permutation index matrix (M permutations)

    Returns
    -------
    navs : (M,)   Normalised Area Value per permutation
    dcos : (M,)   Distance of Centroid from Origin per permutation

    Method
    ------
    NAV = Σ(r_i · r_{i+1}) / N   (sum over N consecutive pairs, cyclic)
    DCO = √(Cx² + Cy²)  where (Cx, Cy) is the polygon centroid computed
          via the shoelace formula.  The regular polygon's center is at the
          origin (0, 0), so DCO is the Euclidean distance from the centroid
          to the center of the ideal polygon.
    """
    N         = r.shape[0]
    ca, sa    = _vertex_angles(N)
    ca_n      = np.roll(ca, -1)     # cos of next-vertex angle
    sa_n      = np.roll(sa, -1)     # sin of next-vertex angle

    rp        = r[perm_idx]                   # (M, N)  metric at each position
    rp_n      = np.roll(rp, -1, axis=1)       # (M, N)  metric at next position

    # ── Normalised Area Value ─────────────────────────────────────────────────
    navs = np.sum(rp * rp_n, axis=1) / N      # (M,)

    # ── Polygon centroid via shoelace ─────────────────────────────────────────
    x,   y   = rp   * ca,   rp   * sa         # current vertex  (M, N)
    x_n, y_n = rp_n * ca_n, rp_n * sa_n       # next vertex     (M, N)

    cross = x * y_n - x_n * y                 # (M, N)  shoelace cross-products
    A     = 0.5 * np.sum(cross, axis=1)        # (M,)    signed polygon area

    safe  = np.abs(A) > 1e-12                  # degenerate-polygon guard
    As    = np.where(safe, A, 1.0)

    Cx = np.where(safe,
                  np.sum((x + x_n) * cross, axis=1) / (6.0 * As), 0.0)
    Cy = np.where(safe,
                  np.sum((y + y_n) * cross, axis=1) / (6.0 * As), 0.0)

    return navs, np.hypot(Cx, Cy)


def _nav_closed_form(r: np.ndarray) -> float:
    """
    Exact closed-form expression for the mean NAV averaged over all
    permutations:

        Av-NAV  =  μ²  −  σ² / (N − 1)

    where  μ  = arithmetic mean of the metrics,
           σ² = (biased) variance  = mean((r − μ)²).

    This is algebraically exact; floating-point error is ~2 × 10⁻¹⁶.
    """
    N = len(r)
    return float(r.mean() ** 2 - r.var() / (N - 1))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Public computation API
# ══════════════════════════════════════════════════════════════════════════════

def compute_gd(metrics,
               n_samples: int = N_SAMPLES,
               seed: int = 42) -> dict:
    """
    Compute Golden Distance for a single classifier.

    Parameters
    ----------
    metrics   : array-like of N floats, N ∈ {3 … 10}
                All values should be normalised to [0, 1].
    n_samples : int
                Number of Monte Carlo draws (used only when N > 8).
                Default: 100 000.
    seed      : int
                RNG seed for Monte Carlo reproducibility.

    Returns
    -------
    dict with the following keys:

      'nav'          float   Mean Normalised Area Value across permutations.
                             Higher → larger polygon area → better overall performance.
      'dco'          float   Mean Distance of Centroid from Origin.
                             Lower → more balanced performance across metrics.
      'gd'           float   Golden Distance = √((Av-NAV − 1)² + Av-DCO²).
                             Lower → closer to the ideal system  →  better.
      'nav_closed'   float   Av-NAV via closed-form formula (consistency check).
      'navs'         ndarray Per-permutation NAV values (for scatter plotting).
      'dcos'         ndarray Per-permutation DCO values (for scatter plotting).
      'exact'        bool    True if full enumeration was used (N ≤ 8).
      'N'            int     Number of metrics.
      'metrics'      ndarray Metric values as a float64 array.
    """
    r = np.asarray(metrics, dtype=np.float64)
    N = int(r.shape[0])

    if not 3 <= N <= 10:
        raise ValueError(
            f"Number of metrics must be between 3 and 10, got {N}.\n"
            f"For N > 10 extend EXACT_LIMIT or increase n_samples.")

    if np.any(r < -0.01) or np.any(r > 1.01):
        warnings.warn(
            "One or more metric values are outside [0, 1].\n"
            "Ensure all metrics are normalised before calling compute_gd().\n"
            "Note: some metrics (e.g. Kappa, MCC) can be negative; "
            "clipping to [0, 1] is left to the caller.",
            stacklevel=2,
        )

    # Build permutation index matrix
    if N <= EXACT_LIMIT:
        perm_idx = np.array(list(permutations(range(N))), dtype=np.intp)
        exact    = True
    else:
        rng      = np.random.default_rng(seed)
        perm_idx = np.stack(
            [rng.permutation(N) for _ in range(n_samples)]
        ).astype(np.intp)
        exact    = False

    navs, dcos = _batch_nav_dco(r, perm_idx)

    nav = float(np.mean(navs))
    dco = float(np.mean(dcos))
    gd  = float(np.hypot(nav - 1.0, dco))

    return dict(
        nav        = nav,
        dco        = dco,
        gd         = gd,
        nav_closed = _nav_closed_form(r),
        navs       = navs,
        dcos       = dcos,
        exact      = exact,
        N          = N,
        metrics    = r,
    )


def evaluate_systems(systems,
                     n_samples: int = N_SAMPLES,
                     seed: int = 42) -> list[dict]:
    """
    Run compute_gd for every classifier in systems.

    Parameters
    ----------
    systems   : list of M classifiers, each a list / array of N metric values
    n_samples : Monte Carlo draws for N > 8
    seed      : RNG seed

    Returns
    -------
    list of M result dicts (same order as input).
    """
    return [compute_gd(s, n_samples=n_samples, seed=seed) for s in systems]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _draw_radar(ax, metrics: np.ndarray, names: list[str],
                title: str = "") -> None:
    """Draw a radar (spider / polygon) chart on the given Axes."""
    N      = len(metrics)
    ca, sa = _vertex_angles(N)

    # Closed arrays (repeat first element to close the polygon)
    m  = np.append(metrics, metrics[0])
    cx = np.append(ca, ca[0])
    cy = np.append(sa, sa[0])

    # ── Background grid ──────────────────────────────────────────────────────
    for rl in np.linspace(0.25, 1.0, 4):
        t = np.linspace(0, 2 * np.pi, 300)
        ax.plot(rl * np.cos(t), rl * np.sin(t),
                color=_C["grid"], lw=0.7, zorder=1)
        ax.text(-0.02, rl + 0.02, f"{rl:.2f}",
                fontsize=7, color="#A0A0A0", ha="right", va="bottom")

    for i in range(N):
        ax.plot([0, ca[i]], [0, sa[i]],
                color=_C["grid"], lw=0.7, zorder=1)

    # ── Ideal regular polygon (all metrics = 1) ───────────────────────────────
    ax.plot(cx, cy, "--", color=_C["reg"], lw=1.5,
            label="Ideal (all = 1.0)", zorder=2)

    # ── Performance polygon ───────────────────────────────────────────────────
    ax.fill(m * cx, m * cy, alpha=0.28, color=_C["poly"], zorder=3)
    ax.plot(m * cx, m * cy, color=_C["poly"], lw=2.2,
            label="Classifier", zorder=3)

    # Vertex dots
    ax.scatter(metrics * ca, metrics * sa,
               s=40, color=_C["poly"], zorder=4)

    # ── Metric labels ─────────────────────────────────────────────────────────
    pad = 0.22
    for i, (name, v) in enumerate(zip(names, metrics)):
        ax.text((1 + pad) * ca[i], (1 + pad) * sa[i],
                f"{name}\n{v:.4f}",
                ha="center", va="center",
                fontsize=10, fontweight="bold", zorder=5)

    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.65, 1.65)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.7)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=6)


def _draw_scatter(ax, result: dict, title: str = "") -> None:
    """Draw the NAV–DCO permutation scatter on the given Axes."""
    navs_all = result["navs"]
    dcos_all = result["dcos"]
    nav      = result["nav"]
    dco      = result["dco"]
    gd       = result["gd"]
    exact    = result["exact"]
    N        = result["N"]

    # Subsample for large permutation sets
    if len(navs_all) > MAX_DOTS:
        sel      = np.random.default_rng(0).choice(len(navs_all),
                                                    MAX_DOTS, replace=False)
        navs_plt = navs_all[sel]
        dcos_plt = dcos_all[sel]
    else:
        navs_plt = navs_all
        dcos_plt = dcos_all

    n_perms = factorial(N) if exact else f"~{len(navs_all):,}"
    method  = "exact" if exact else "Monte Carlo"

    ax.scatter(navs_plt, dcos_plt, s=18, alpha=0.50,
               color=_C["cloud"], label=f"Permutations ({n_perms}, {method})",
               zorder=2)
    ax.scatter([nav], [dco], s=150, color=_C["mean"], zorder=5,
               label=f"Mean  ({nav:.4f},  {dco:.4f})")
    ax.scatter([1.0], [0.0], s=200, color=_C["ideal"],
               marker="*", zorder=5, label="Ideal  (1, 0)")

    # GD connecting line
    ax.plot([nav, 1.0], [dco, 0.0],
            "--", color=_C["line"], lw=1.8, zorder=4)

    # GD label (placed near the midpoint, offset perpendicular to the line)
    mid_x = (nav + 1.0) / 2
    mid_y = (dco + 0.0) / 2
    dy    = max(dcos_all.max() * 0.08, 0.008)
    ax.text(mid_x, mid_y + dy,
            f"GD = {gd:.4f}",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=_C["line"], zorder=6)

    ax.set_xlabel("Normalised Area Value (NAV)", fontsize=11)
    ax.set_ylabel("Distance of Centroid from Origin (DCO)", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.75, loc="upper left")
    ax.grid(True, lw=0.5, alpha=0.5)
    ax.set_facecolor(_C["bg"])
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — High-level figure generators
# ══════════════════════════════════════════════════════════════════════════════

def plot_system(result: dict,
                metric_names: list[str],
                system_name: str = "",
                save_path: str | None = None) -> plt.Figure:
    """
    Two-panel figure for a single classifier.

    Left panel  – Radar chart (performance polygon vs ideal polygon).
    Right panel – NAV–DCO scatter plot with GD annotation.

    Parameters
    ----------
    result       : dict returned by compute_gd()
    metric_names : list of N metric label strings
    system_name  : title for the figure
    save_path    : if given, figure is saved to this path (PNG, 180 dpi)

    Returns
    -------
    matplotlib.figure.Figure
    """
    with plt.rc_context({"font.family": "DejaVu Sans",
                         "axes.spines.top": False,
                         "axes.spines.right": False}):
        fig, (ax_r, ax_s) = plt.subplots(
            1, 2, figsize=(14, 6), facecolor=_C["bg"])

        suptitle = (system_name or "Golden Distance Analysis")
        gd_str   = f"GD = {result['gd']:.4f}"
        fig.suptitle(f"{suptitle}   |   {gd_str}",
                     fontsize=14, fontweight="bold")

        _draw_radar(ax_r, result["metrics"], metric_names,
                    title="Performance Polygon")
        _draw_scatter(ax_s, result, title="NAV – DCO Space")

        fig.subplots_adjust(left=0.04, right=0.97, top=0.92,
                            bottom=0.10, wspace=0.28)
        if save_path:
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


def plot_ranking(results: list[dict],
                 system_names: list[str],
                 save_path: str | None = None) -> plt.Figure:
    """
    Summary figure: ranked bar chart of GD values + result table.

    The bar chart uses gold / silver / bronze colours for the top-3 systems
    (ranked by lowest GD, i.e. best performance).

    Parameters
    ----------
    results      : list of dicts returned by evaluate_systems()
    system_names : list of M system name strings
    save_path    : optional PNG save path

    Returns
    -------
    matplotlib.figure.Figure
    """
    M    = len(results)
    gds  = np.array([r["gd"]  for r in results])
    navs = np.array([r["nav"] for r in results])
    dcos = np.array([r["dco"] for r in results])
    order = np.argsort(gds)            # best → worst

    s_gd    = gds[order]
    s_nav   = navs[order]
    s_dco   = dcos[order]
    s_names = [system_names[i] for i in order]

    bar_colours = (
        [_C["gold"], _C["silver"], _C["bronze"]]
        + [_C["bar"]] * M
    )[:M]

    with plt.rc_context({"font.family": "DejaVu Sans",
                         "axes.spines.top": False,
                         "axes.spines.right": False}):
        fig = plt.figure(figsize=(max(9, M * 1.5), 8), facecolor=_C["bg"])
        gs  = GridSpec(2, 1, height_ratios=[3, 1.3], hspace=0.5)
        ax_b = fig.add_subplot(gs[0])
        ax_t = fig.add_subplot(gs[1])

        # ── Bar chart ─────────────────────────────────────────────────────────
        x_pos = np.arange(M)
        bars  = ax_b.bar(x_pos, s_gd, color=bar_colours,
                         edgecolor="white", linewidth=0.8, width=0.55)

        for bar, gd_val in zip(bars, s_gd):
            ax_b.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s_gd.max() * 0.012,
                f"{gd_val:.4f}",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold")

        ax_b.set_xticks(x_pos)
        ax_b.set_xticklabels(s_names, rotation=22, ha="right", fontsize=10)
        ax_b.set_ylabel("Golden Distance", fontsize=11)
        ax_b.set_title(
            "Classifier Ranking  —  ↓ Lower GD = Better Performance",
            fontsize=13, fontweight="bold")
        ax_b.set_ylim(0, s_gd.max() * 1.22)
        ax_b.grid(axis="y", lw=0.5, alpha=0.5)
        ax_b.set_facecolor(_C["bg"])

        # ── Result table ──────────────────────────────────────────────────────
        ax_t.axis("off")
        col_hdrs = ["Classifier", "Rank", "Av-NAV", "Av-DCO", "Golden Distance"]
        rows = [
            [s_names[i],
             f"# {i + 1}",
             f"{s_nav[i]:.4f}",
             f"{s_dco[i]:.4f}",
             f"{s_gd[i]:.4f}"]
            for i in range(M)
        ]
        tbl = ax_t.table(cellText=rows, colLabels=col_hdrs,
                         loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.7)

        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor("#D5D8DC")
            if row == 0:
                cell.set_facecolor(_C["hdr"])
                cell.set_text_props(color="white", fontweight="bold")
            elif row == 1:
                cell.set_facecolor("#FDEBD0")   # highlight best
            elif row % 2 == 0:
                cell.set_facecolor("#EBF5FB")
            else:
                cell.set_facecolor("white")

        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.97,
                            hspace=0.55)
        if save_path:
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_golden_distance(
        systems,
        metric_names,
        system_names=None,
        n_samples: int = N_SAMPLES,
        seed: int = 42,
        save_dir: str | None = None,
        show_plots: bool = True,
        verbose: bool = True,
) -> list[dict]:
    """
    Compute Golden Distance for all classifiers and produce all figures.

    This is the main entry point for typical usage.

    Parameters
    ----------
    systems      : list of M classifiers, each a list of N metric values.
                   N must be the same for all classifiers.  N ∈ {3 … 10}.
    metric_names : list of N strings labelling each metric column.
    system_names : list of M strings naming each classifier row.
                   Default: "System 1", "System 2", …
    n_samples    : Monte Carlo draws for N > 8.  Default: 100 000.
    seed         : RNG seed.  Default: 42.
    save_dir     : directory path.  If given, all figures are saved as PNG
                   files here instead of (or in addition to) being shown.
    show_plots   : show figures on screen (set False for batch/headless runs).
    verbose      : print the result table to stdout.

    Returns
    -------
    list of M result dicts — see compute_gd() for the full key listing.

    Examples
    --------
    >>> results = run_golden_distance(
    ...     systems=[
    ...         [0.9444, 0.9537, 0.9533, 0.9488],
    ...         [0.9259, 0.8889, 0.8929, 0.9091],
    ...     ],
    ...     metric_names=["ACC", "Sen", "Spec", "F1"],
    ...     system_names=["Method A", "Method B"],
    ... )
    >>> print(results[0]["gd"])
    """
    systems      = [np.asarray(s, dtype=float) for s in systems]
    M            = len(systems)
    N            = int(systems[0].shape[0])
    metric_names = list(metric_names)

    if system_names is None:
        system_names = [f"System {i + 1}" for i in range(M)]
    else:
        system_names = list(system_names)

    if len(metric_names) != N:
        raise ValueError(
            f"Expected {N} metric names (one per column), "
            f"got {len(metric_names)}.")
    if len(system_names) != M:
        raise ValueError(
            f"Expected {M} system names (one per row), "
            f"got {len(system_names)}.")
    if save_dir and not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # ── Compute ───────────────────────────────────────────────────────────────
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  Golden Distance  |  {M} classifier(s)  ×  {N} metric(s)")
    print(f"  N ≤ {EXACT_LIMIT}: exact  |  N > {EXACT_LIMIT}: Monte Carlo "
          f"({n_samples:,} samples)")
    print(sep)

    results = evaluate_systems(systems, n_samples=n_samples, seed=seed)

    # ── Console table ─────────────────────────────────────────────────────────
    if verbose:
        w = max(max(len(n) for n in system_names), 11)
        print(f"\n  {'Classifier':<{w}}  {'Av-NAV':>8}  {'Av-DCO':>8}"
              f"  {'GD':>8}  NAV(closed)  Method")
        print(f"  {'─'*w}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*11}  {'─'*12}")
        for name, res in zip(system_names, results):
            method = "exact" if res["exact"] else f"MC {n_samples:,}"
            print(
                f"  {name:<{w}}  {res['nav']:>8.4f}  {res['dco']:>8.4f}"
                f"  {res['gd']:>8.4f}  {res['nav_closed']:>11.4f}  {method}"
            )
        best_i  = int(np.argmin([r["gd"] for r in results]))
        worst_i = int(np.argmax([r["gd"] for r in results]))
        print(
            f"\n  ✓  Best  : {system_names[best_i]}"
            f"  (GD = {results[best_i]['gd']:.4f})"
        )
        print(
            f"     Worst : {system_names[worst_i]}"
            f"  (GD = {results[worst_i]['gd']:.4f})"
        )
        print()

    # ── Per-system plots ──────────────────────────────────────────────────────
    for i, (name, res) in enumerate(zip(system_names, results)):
        sp = None
        if save_dir:
            safe = (name.replace(" ", "_")
                       .replace("/", "-")
                       .replace("\\", "-"))
            sp = os.path.join(save_dir, f"gd_{i+1:02d}_{safe}.png")

        fig = plot_system(res, metric_names, name, save_path=sp)
        if sp:
            print(f"  Saved: {sp}")
        if show_plots:
            plt.show()
        plt.close(fig)

    # ── Ranking figure ────────────────────────────────────────────────────────
    sp2 = os.path.join(save_dir, "gd_ranking.png") if save_dir else None
    fig2 = plot_ranking(results, system_names, save_path=sp2)
    if sp2:
        print(f"  Saved: {sp2}")
    if show_plots:
        plt.show()
    plt.close(fig2)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — Standalone demo
#  Run directly:  python golden_distance.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Example 1: 4-metric case (data from the paper) ───────────────────────
    print("\n▶  Example 1 — 4 metrics (paper dataset)")
    run_golden_distance(
        systems=[
            [0.9444, 0.9537, 0.9533, 0.9488],
            [0.9259, 0.8889, 0.8929, 0.9091],
            [0.9722, 0.8611, 0.8750, 0.9211],
            [0.9167, 0.8611, 0.8684, 0.8919],
            [0.8981, 0.8981, 0.8981, 0.8981],
            [0.9352, 0.8796, 0.8860, 0.9099],
        ],
        metric_names=["ACC", "Sensitivity", "Specificity", "F1"],
        system_names=[f"Classifier {i}" for i in range(1, 7)],
    )

    # ── Example 2: 6-metric case ─────────────────────────────────────────────
    print("\n▶  Example 2 — 6 metrics")
    run_golden_distance(
        systems=[
            [0.85, 0.88, 0.82, 0.86, 0.79, 0.83],
            [0.91, 0.87, 0.93, 0.89, 0.85, 0.88],
            [0.78, 0.92, 0.75, 0.84, 0.81, 0.86],
            [0.95, 0.71, 0.97, 0.82, 0.90, 0.76],
        ],
        metric_names=["ACC", "Sensitivity", "Specificity", "F1", "AUC", "JI"],
        system_names=["SVM", "k-NN", "LDA", "Random Forest"],
    )

    # ── Example 3: 8-metric case (exact enumeration upper boundary) ──────────
    print("\n▶  Example 3 — 8 metrics (N = EXACT_LIMIT)")
    run_golden_distance(
        systems=[
            [0.90, 0.88, 0.91, 0.89, 0.87, 0.92, 0.86, 0.90],
            [0.85, 0.83, 0.86, 0.84, 0.82, 0.87, 0.81, 0.85],
            [0.78, 0.95, 0.74, 0.82, 0.76, 0.88, 0.79, 0.83],
        ],
        metric_names=["ACC", "Sen", "Spec", "F1",
                      "AUC", "JI", "MCC", "Kappa"],
        system_names=["Deep CNN", "ResNet-50", "VGG-16"],
    )

    # ── Example 4: 10-metric case (Monte Carlo) ───────────────────────────────
    print("\n▶  Example 4 — 10 metrics (Monte Carlo sampling)")
    run_golden_distance(
        systems=[
            [0.91, 0.89, 0.92, 0.90, 0.88, 0.93, 0.87, 0.91, 0.85, 0.90],
            [0.88, 0.86, 0.89, 0.87, 0.85, 0.90, 0.84, 0.88, 0.82, 0.87],
        ],
        metric_names=["M1", "M2", "M3", "M4", "M5",
                      "M6", "M7", "M8", "M9", "M10"],
        system_names=["System A", "System B"],
        n_samples=50_000,    # fewer samples for speed in demo
    )
