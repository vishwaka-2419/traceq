#!/usr/bin/env python3
"""Regenerate every manuscript and supplement figure.

Reads  data/results/study.json, theory.json, robustness.json and data/traces/*.csv
Writes figures/*.pdf and figures/*.png

House rules: no in-figure titles (captions carry the message), panel letters
only, and no legend or annotation placed over data.
"""
from __future__ import annotations
import csv, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

_have = {f.name for f in fm.fontManager.ttflist}
# TrueType families only: CFF-flavoured OpenType fonts embed badly with pdf.fonttype 42.
FAMILY = next((f for f in ("Arial", "Liberation Sans", "DejaVu Sans")
               if f in _have), "DejaVu Sans")

# ----------------------------------------------------------------- style
INK, INK2, RULE = "#152A38", "#4C6373", "#D6DEE3"
C_JW, C_TT = "#1F5C99", "#D08B12"
C_DET, C_WARN, C_OK = "#6B7F8C", "#B3322C", "#2E7D5B"

plt.rcParams.update({
    "font.family": FAMILY, "font.size": 8.0,
    "axes.labelsize": 8.2, "xtick.labelsize": 7.6, "ytick.labelsize": 7.6, "legend.fontsize": 7.4,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0, "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
    "lines.linewidth": 1.4, "lines.markersize": 4.0,
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.columnspacing": 1.3,
    "legend.handletextpad": 0.55, "legend.borderaxespad": 0.2, "legend.labelspacing": 0.35,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "custom", "mathtext.rm": FAMILY, "mathtext.it": FAMILY + ":italic",
    "mathtext.bf": FAMILY + ":bold", "mathtext.fallback": "cm",
})
W2 = 6.45          # full text width in inches


def tidy(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(True, axis=grid, color=RULE, lw=0.6, alpha=0.9)
        ax.set_axisbelow(True)
    return ax


def panel(ax, letter, dx=-0.16, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9.6, fontweight="bold",
            va="bottom", ha="left", color=INK)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=400)
    plt.close(fig)
    print("  wrote", name)


# ------------------------------------------------------------------ data
DATA = ROOT / "data"
study = json.load(open(DATA / "results" / "study.json"))
if study.get("quick", False):
    raise RuntimeError("Refusing to build manuscript figures from smoke-run data")
RES = {r["id"]: r for r in study["results"]}
EXACT = study["exact"]
THEORY = json.load(open(DATA / "results" / "theory.json"))
ROBUST = json.load(open(DATA / "results" / "robustness.json"))
PRED = {r["id"]: r for r in THEORY["predictor"]}


def excess(rid):
    return RES[rid]["rate_excess_pct"]


def ci(rid):
    s, t0 = RES[rid]["stats"], RES[rid]["trate"]
    m = (s["mean"] / t0 - 1) * 100
    return m, m - (s["mean_ci_low"] / t0 - 1) * 100, (s["mean_ci_high"] / t0 - 1) * 100 - m


def trace(name):
    rows = list(csv.reader(open(DATA / "traces" / f"{name}.csv")))
    return np.array([int(r[1]) for r in rows[1:]], dtype=int)


# =====================================================================
# FIGURE 1 -- model and interface
# =====================================================================
def fig_model():
    fig = plt.figure(figsize=(W2, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.06)
    ax = fig.add_subplot(gs[0]); ax.set_axis_off()
    ax.set_xlim(0, 112); ax.set_ylim(0, 80)

    def rbox(a, x, y, w, h, fc="white", ec=INK2, lw=0.9):
        a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                                   fc=fc, ec=ec, lw=lw, zorder=3))

    def arrow(a, x1, y1, x2, y2, c=INK2, ls="-", lw=1.0, ms=8):
        a.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                                    color=c, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0, zorder=2))

    # producers
    ax.text(1, 71.5, "$K$ producers", fontsize=7.9, fontweight="bold", color=INK, va="center")
    for y, lab, col in ((55, "cultivating", INK2), (42, "cultivating", INK2),
                        (26, "blocked: holds an\naccepted state", C_WARN)):
        h = 11 if col is INK2 else 13
        rbox(ax, 1, y, 30, h, fc="#F4F8FA", ec=RULE if col is INK2 else C_WARN)
        ax.text(16, y + h / 2, lab, ha="center", va="center", fontsize=6.6, color=col,
                zorder=4, linespacing=1.35)
    ax.text(16, 17.5, r"latency $L=aG$,  $G\sim\mathrm{Geom}(p)$", ha="center", fontsize=6.7, color=INK)
    ax.text(16, 11.0, r"mean $\mu=a/p$", ha="center", fontsize=6.7, color=INK)

    # buffer: title, subtitle, then slots well below the text
    rbox(ax, 42, 26, 24, 40, fc="#EDF3F6")
    ax.text(54, 60.5, "buffer", ha="center", va="center", fontsize=7.9, fontweight="bold", zorder=4)
    ax.text(54, 53.5, "FIFO", ha="center", va="center", fontsize=6.7, color=INK2, zorder=4)
    ax.text(54, 47.5, "capacity $B$", ha="center", va="center", fontsize=6.7, color=INK2, zorder=4)
    for i in range(4):
        ax.add_patch(Rectangle((45.0 + i * 4.6, 31.0), 3.6, 7.5,
                               fc=C_OK if i < 3 else "white", ec=INK2, lw=0.6, zorder=5))

    # consumer: three short lines inside a wide box
    rbox(ax, 77, 26, 34, 40, fc="#F4F8FA")
    ax.text(94, 60.5, "consumer", ha="center", va="center", fontsize=7.9, fontweight="bold", zorder=4)
    ax.text(94, 51.5, "ordered trace", ha="center", va="center", fontsize=6.7, color=INK2, zorder=4)
    ax.text(94, 44.5, r"$m_1,\,m_2,\,\ldots,\,m_S$", ha="center", va="center", fontsize=7.0, color=INK, zorder=4)
    ax.text(94, 37.0, "atomic demand", ha="center", va="center", fontsize=6.7, color=INK2, zorder=4)
    ax.text(94, 31.0, r"step time $\tau$", ha="center", va="center", fontsize=6.7, color=INK2, zorder=4)

    for y in (60.5, 47.5, 32.5):
        arrow(ax, 31, y, 42, 46)
    arrow(ax, 66, 46, 77, 46)
    ax.text(54, 20.5, "full buffer blocks\nthe producer", ha="center", va="center", fontsize=6.6,
            color=C_WARN, linespacing=1.3)
    ax.text(94, 20.5, "stall while\nstock $< m_s$", ha="center", va="center", fontsize=6.6,
            color=C_WARN, linespacing=1.3)
    ax.text(56, 3.0, "at most $B+K$ accepted states exist at once", ha="center", fontsize=6.9,
            color=INK2, style="italic")
    ax.text(-1, 80, "a", fontsize=9.6, fontweight="bold", color=INK, va="top")

    ax2 = fig.add_subplot(gs[1]); ax2.set_axis_off()
    ax2.set_xlim(0, 64); ax2.set_ylim(0, 80)
    rungs = ((57.5, "#FCF5EA", C_TT, "rate", r"$N_T,\ S,\ \mu,\ K$",
              r"point estimate $T_{\mathrm{rate}}$"),
             (31.5, "#F3F0F7", "#6B4E9B", "histogram", "adds peak, mean, all step counts",
              r"interval $[H^-,H^+]$,  $H^+<2H^-$"),
             (5.5, "#EDF3F6", C_JW, "ordered trace", r"adds the order of $m_1,\ldots,m_S$",
              r"$T^{\mathrm{det}}$ exactly, in one pass"))
    for y, fc, ec, head, what, gives in rungs:
        rbox(ax2, 2, y, 61, 19.5, fc=fc, ec=ec)
        ax2.text(5, y + 15.2, head, fontsize=7.9, fontweight="bold", color=INK, va="center")
        ax2.text(5, y + 9.6, what, fontsize=6.7, color=INK2, va="center")
        ax2.text(5, y + 3.9, gives, fontsize=7.0, color=INK, va="center")
    for y in (57.5, 31.5):
        arrow(ax2, 32.5, y - 0.3, 32.5, y - 6.2, c=INK2, lw=1.0, ms=7)
    ax2.text(35.5, 54.2, "not enough", fontsize=6.5, color=C_WARN, va="center", fontweight="bold")
    ax2.text(35.5, 28.2, "error floor 1/3", fontsize=6.5, color=C_WARN, va="center", fontweight="bold")
    ax2.text(-1, 80, "b", fontsize=9.6, fontweight="bold", color=INK, va="top")
    save(fig, "fig1_model")


# =====================================================================
# FIGURE 2 -- the exact separation family
# =====================================================================
def fig_exact():
    w, c, r = 4, 2, 3
    spread = np.tile(np.array([w] + [0] * (w - 1)), c * r)
    cluster = np.tile(np.array([w] * c + [0] * (c * (w - 1))), r)
    fig = plt.figure(figsize=(W2, 2.55))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.46, hspace=0.35)
    axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[1, 0], sharex=axa)
    for ax, m, col, lab in ((axa, spread, C_JW, "spread, $T=24$"), (axb, cluster, C_WARN, "clustered, $T=31$")):
        ax.bar(np.arange(1, len(m) + 1), m, width=0.62, color=col, lw=0)
        ax.set_ylim(0, 7.2); ax.set_yticks([0, 4]); tidy(ax, grid=None)
        ax.set_ylabel("$m_s$", labelpad=1)
        ax.text(0.99, 0.97, lab, transform=ax.transAxes, ha="right", va="top", fontsize=7.3, color=col)
    axb.set_xlabel("step $s$", labelpad=1); axa.tick_params(labelbottom=False)
    panel(axa, "a", dx=-0.17, dy=1.06)

    ax = fig.add_subplot(gs[:, 1]); tidy(ax)
    reps = sorted({e["repeats"] for e in EXACT})
    for wid, col, mk in ((3, C_JW, "o"), (4, C_WARN, "s")):
        xs = np.array([e["repeats"] for e in EXACT if e["w"] == wid])
        ys = np.array([e["excess_pct"] for e in EXACT if e["w"] == wid]); o = np.argsort(xs)
        ax.plot(xs[o], ys[o], mk + "-", color=col, mfc="white", mew=1.3, label=f"$w={wid}$")
        ax.axhline((3 * wid - 2) / (2 * wid) * 100 - 100, color=col, lw=0.8, ls=(0, (4, 2.5)), alpha=0.8)
    ax.set_xscale("log", base=2); ax.set_xticks(reps); ax.set_xticklabels([str(x) for x in reps])
    ax.set_ylim(14, 46)
    ax.set_xlabel("repetitions $r$  (cluster length $c=2$)", labelpad=1)
    ax.set_ylabel("runtime excess, clustered vs spread (%)", labelpad=2)
    ax.legend(loc="upper right", ncol=2)
    panel(ax, "b", dx=-0.27, dy=1.02)

    ax = fig.add_subplot(gs[:, 2]); tidy(ax)
    fam = [f for f in THEORY["family"] if f["r"] == 64]
    for wid, col, mk in ((3, C_JW, "o"), (4, C_WARN, "s"), (8, C_OK, "^"), (16, INK2, "D")):
        rows = sorted((f for f in fam if f["w"] == wid), key=lambda f: f["c"])
        ax.plot([f["c"] for f in rows], [f["excess_pct"] for f in rows], mk + "-", color=col,
                mfc="white", mew=1.2, ms=3.6, label=f"$w={wid}$")
    ax.axhline(100, color=INK, lw=0.9, ls=(0, (4, 2.5)))
    ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16, 32]); ax.set_xticklabels(["2", "4", "8", "16", "32"])
    ax.set_ylim(0, 135); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("cluster length $c$  ($r=64$)", labelpad=1)
    ax.set_ylabel("runtime excess, clustered vs spread (%)", labelpad=2)
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.0))
    panel(ax, "c", dx=-0.27, dy=1.02)
    save(fig, "fig2_exact")


# =====================================================================
# FIGURE 3 -- the reconstructed Hubbard demand traces
# =====================================================================
def fig_traces():
    jw, tt = trace("hubbard_6x4_JW"), trace("hubbard_6x4_TT")

    def roll(m, w=207):
        return np.convolve(m.astype(float), np.ones(w) / w, mode="same")

    fig = plt.figure(figsize=(W2, 2.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 1.5, 1.15], wspace=0.40)
    ax = fig.add_subplot(gs[0]); tidy(ax)
    ax.add_patch(Rectangle((3000, 0.0), 700, 2.78, fc=INK, alpha=0.08, lw=0, zorder=0))
    for m, col, lab in ((jw, C_JW, "Jordan-Wigner"), (tt, C_TT, "ternary tree")):
        ax.plot(np.arange(len(m)), roll(m), color=col, lw=1.1, label=lab)
    ax.axhline(1.0, color=INK2, lw=0.7, ls=(0, (3, 3)))
    ax.set_xlim(0, 8100); ax.set_ylim(0.6, 3.35); ax.set_yticks([1.0, 1.5, 2.0, 2.5])
    ax.set_xlabel("scheduled step $s$", labelpad=1)
    ax.set_ylabel("demand, 207-step mean", labelpad=2)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    panel(ax, "a", dx=-0.19)

    ax = fig.add_subplot(gs[1]); tidy(ax, grid=None)
    lo, hi = 3000, 3700
    ax.step(np.arange(lo, hi), jw[lo:hi], where="mid", color=C_JW, lw=0.9)
    ax.step(np.arange(lo, hi), tt[lo:hi] + 6, where="mid", color=C_TT, lw=0.9)
    ax.set_ylim(0, 11.6); ax.set_yticks([0, 2, 4, 6, 8, 10]); ax.set_yticklabels(["0", "2", "4", "0", "2", "4"])
    ax.set_xticks([3000, 3350, 3700])
    ax.set_xlabel("step $s$", labelpad=1); ax.set_ylabel("$m_s$", labelpad=2)
    ax.text(3008, 11.3, "ternary tree", ha="left", va="top", fontsize=7.0, color=C_TT)
    ax.text(3008, 5.3, "Jordan-Wigner", ha="left", va="top", fontsize=7.0, color=C_JW)
    panel(ax, "b", dx=-0.22)

    ax = fig.add_subplot(gs[2]); tidy(ax)
    hj = RES["baseline_JW"]["signature"]["histogram"]; ht = RES["baseline_TT"]["signature"]["histogram"]
    keys = sorted({int(k) for k in list(hj) + list(ht)}); x = np.arange(len(keys))
    ax.bar(x - 0.2, [hj.get(str(k), 0) for k in keys], width=0.38, color=C_JW, lw=0)
    ax.bar(x + 0.2, [ht.get(str(k), 0) for k in keys], width=0.38, color=C_TT, lw=0)
    ax.set_xticks(x); ax.set_xticklabels(keys)
    ax.set_xlabel("demand $m_s$", labelpad=1); ax.set_ylabel("steps", labelpad=2)
    panel(ax, "c", dx=-0.42)
    save(fig, "fig3_traces")


# =====================================================================
# FIGURE 4 -- capacity sweep with deterministic control and closed form
# =====================================================================
def fig_capacity():
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.45), gridspec_kw={"wspace": 0.30, "width_ratios": [1.25, 1.0]})
    ax = axes[0]; tidy(ax)
    env = THEORY["envelope"]
    for mp, col in (("JW", C_JW), ("TT", C_TT)):
        rows = [e for e in env if e["mapping"] == mp]
        ax.plot([e["K"] for e in rows], [e["det_compiled"] for e in rows], "-", color=col, lw=1.0, alpha=0.55)
        ks, ms, los, his = [], [], [], []
        for k in range(6, 25):
            m, lo, hi = ci(f"capacity_{mp}_{k}")
            ks.append(k); ms.append(m); los.append(lo); his.append(hi)
        ax.errorbar(ks, ms, yerr=[los, his], fmt="o", color=col, mfc="white", mew=1.2, ms=3.6,
                    capsize=1.6, elinewidth=0.8)
    ax.axvline(10, color=C_JW, lw=0.7, ls=(0, (3, 3)), alpha=0.6)
    ax.axvline(11, color=C_TT, lw=0.7, ls=(0, (3, 3)), alpha=0.6)
    ax.set_xlabel("production patches $K$", labelpad=1)
    ax.set_ylabel(r"excess over $T_{\mathrm{rate}}$ (%)", labelpad=2)
    ax.set_xticks(range(6, 25, 3)); ax.set_ylim(-0.8, 22)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C_JW, mfc="white", mew=1.2),
                       Line2D([], [], marker="o", ls="", color=C_TT, mfc="white", mew=1.2),
                       Line2D([], [], color=C_DET, lw=1.0)],
              labels=["JW, simulated mean", "TT, simulated mean", "closed form, deterministic"],
              loc="upper right")
    panel(ax, "a")

    ax = axes[1]; tidy(ax)
    geo = [excess("baseline_JW"), excess("baseline_TT")]
    det = [excess("law_JW_deterministic"), excess("law_TT_deterministic")]
    x = np.arange(2)
    ax.bar(x - 0.19, geo, width=0.34, color=[C_JW, C_TT], lw=0)
    ax.bar(x + 0.19, det, width=0.34, color=[C_JW, C_TT], lw=0, alpha=0.42, hatch="////", edgecolor="white")
    for i in range(2):
        ax.text(x[i] - 0.19, geo[i] + 0.45, f"{geo[i]:.2f}", ha="center", fontsize=7.0, color=INK)
        ax.text(x[i] + 0.19, det[i] + 0.45, f"{det[i]:.2f}", ha="center", fontsize=7.0, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Jordan-Wigner\n{det[0] / geo[0] * 100:.0f}% retained",
                        f"ternary tree\n{det[1] / geo[1] * 100:.0f}% retained"], fontsize=7.2)
    ax.set_ylim(0, 30)
    ax.set_ylabel(r"excess at $K_{\mathrm{avg}}$, $B=4$ (%)", labelpad=2)
    ax.legend(handles=[Patch(fc=C_DET, lw=0), Patch(fc=C_DET, alpha=0.42, hatch="////", edgecolor="white", lw=0)],
              labels=["geometric service", r"deterministic, same $\mu$"], loc="upper center", ncol=1)
    panel(ax, "b", dx=-0.2)
    save(fig, "fig4_capacity")


# =====================================================================
# FIGURE 6 -- demand order under a fixed complete histogram
# =====================================================================
def fig_order():
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.4), gridspec_kw={"wspace": 0.3, "width_ratios": [1.2, 1.0]})
    ax = axes[0]; tidy(ax)
    order = [("order_interleaved", "interleaved"), ("order_shuffled", "shuffled"),
             ("order_compiled", "compiled"), ("order_clustered", "clustered")]
    vals = [excess(k) for k, _ in order]; dets = [PRED[k]["det"] for k, _ in order]
    x = np.arange(len(order))
    ax.bar(x, vals, width=0.6, color=[C_DET, C_DET, C_TT, C_WARN], lw=0)
    ax.plot(x, dets, ls="", marker="_", ms=17, mew=1.8, color=INK, zorder=5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.4, f"{v:.2f}", ha="center", fontsize=7.1, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in order], fontsize=7.3)
    ax.set_ylim(0, 22)
    ax.set_ylabel(r"excess over $T_{\mathrm{rate}}$ (%)", labelpad=2)
    ax.legend(handles=[Patch(fc=C_DET, lw=0), Line2D([], [], ls="", marker="_", ms=12, mew=1.8, color=INK)],
              labels=["simulated mean", "closed form (deterministic)"], loc="upper left")
    panel(ax, "a")

    ax = axes[1]; tidy(ax)
    ens = sorted(excess(f"shuffle_ensemble_{i}") for i in range(32))
    ax.hist(ens, bins=np.arange(5.05, 5.43, 0.03), color=C_DET, alpha=0.85, lw=0)
    ax.set_xlim(5.03, 5.43); ax.set_ylim(0, 9)
    ax.set_xlabel(r"mean excess over $T_{\mathrm{rate}}$ (%)", labelpad=1)
    ax.set_ylabel("random permutations", labelpad=2)
    panel(ax, "b", dx=-0.19)
    save(fig, "fig6_order")


# =====================================================================
# FIGURE 8 -- storage: runtime against area and residency
# =====================================================================
def fig_buffer():
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.4), gridspec_kw={"wspace": 0.32})
    ax = axes[0]; tidy(ax)
    BS = (4, 8, 12, 16, 24, 32, 64)
    for mp, col, kb in (("JW", C_JW, 10), ("TT", C_TT, 11)):
        xs = [kb + b for b in BS]
        ax.plot(xs, [PRED[f"buffer_{mp}_{b}"]["det"] for b in BS], "-", color=col, lw=1.0, alpha=0.55)
        ax.plot(xs, [excess(f"buffer_{mp}_{b}") for b in BS], "o", color=col, mfc="white", mew=1.2)
    ax.set_xlabel("normalised supply-side area $K+B$", labelpad=1)
    ax.set_ylabel(r"excess over $T_{\mathrm{rate}}$ (%)", labelpad=2)
    ax.set_ylim(6.5, 23.5); ax.set_xlim(8, 82)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C_JW, mfc="white", mew=1.2),
                       Line2D([], [], marker="o", ls="", color=C_TT, mfc="white", mew=1.2),
                       Line2D([], [], color=C_DET, lw=1.0)],
              labels=["JW, $K=10$, simulated", "TT, $K=11$, simulated", "closed form"], loc="upper right", ncol=1)
    panel(ax, "a")

    ax = axes[1]; tidy(ax)
    for mp, col in (("JW", C_JW), ("TT", C_TT)):
        ax.plot(BS, [RES[f"buffer_{mp}_{b}"]["mean_residency"] for b in BS], "s-", color=col,
                mfc="white", mew=1.2, label=mp)
    ax.set_xscale("log", base=2); ax.set_xticks([4, 8, 16, 32, 64]); ax.set_xticklabels(["4", "8", "16", "32", "64"])
    ax.set_xlabel("external storage $B$", labelpad=1)
    ax.set_ylabel("mean residency (cycles)", labelpad=2)
    ax.legend(loc="upper left")
    panel(ax, "b", dx=-0.2)
    save(fig, "fig8_buffer")


# =====================================================================
# FIGURE 10 -- deadline certificate
# =====================================================================
def fig_validation():
    rows = [(k, RES[k]) for k in RES if k.startswith("validation_")]
    rows.sort(key=lambda kv: (kv[0].split("_")[1], int(kv[0].split("_")[3])))
    fig, ax = plt.subplots(figsize=(W2, 2.2)); tidy(ax)
    labs, ucb, obs, cols = [], [], [], []
    for k, r in rows:
        _, mp, K, B = k.split("_")
        labs.append(f"{mp}\n$K$={K}, $B$={B}")
        ucb.append(r["validation"]["violation_ucb_family"]); obs.append(r["stats"]["violation_rate"])
        cols.append(C_JW if mp == "JW" else C_TT)
    x = np.arange(len(labs))
    ax.bar(x, ucb, width=0.56, color=cols, lw=0, alpha=0.92)
    ax.plot(x, obs, "o", color=INK, ms=3.8, zorder=5)
    ax.axhline(0.05, color=C_WARN, lw=1.1, ls=(0, (4, 2.4)))
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6.9)
    ax.set_ylim(0, 0.068); ax.set_yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])
    ax.set_ylabel(r"$\Pr\{T>1.02\,T_0\}$", labelpad=2)
    ax.legend(handles=[Patch(fc=C_DET, lw=0), Line2D([], [], marker="o", ls="", color=INK, ms=3.8),
                       Line2D([], [], color=C_WARN, lw=1.1, ls=(0, (4, 2.4)))],
              labels=["simultaneous 95% upper bound", "observed violation rate", r"target $\delta=0.05$"],
              loc="upper center", ncol=3)
    save(fig, "fig10_validation")


# =====================================================================
# FIGURE 5 -- the closed form against the stochastic study
# =====================================================================
def fig_predictor():
    fig = plt.figure(figsize=(W2, 2.95))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.32], wspace=0.30, hspace=0.30)
    ax = fig.add_subplot(gs[:, 0]); tidy(ax, grid="both")
    decl = [r for r in THEORY["predictor"] if r["family"] == "shuffle" or r["id"] in ("order_shuffled", "order_interleaved")]
    stru = [r for r in THEORY["predictor"] if r not in decl]
    ax.plot([0, 22], [0, 22], color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.plot([r["det"] for r in stru], [r["sim"] for r in stru], "o", color=C_JW, mfc="white", mew=0.9, ms=3.2,
            label=f"compiled or clustered ({len(stru)})", zorder=3)
    ax.plot([r["det"] for r in decl], [r["sim"] for r in decl], "s", color=C_WARN, mfc="white", mew=0.9, ms=3.2,
            label=f"shuffled or interleaved ({len(decl)})", zorder=4)
    ax.set_xlim(-0.8, 22); ax.set_ylim(-0.8, 28.5); ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_xlabel("closed-form excess (%)", labelpad=1)
    ax.set_ylabel("simulated mean excess (%)", labelpad=2)
    ax.legend(loc="upper left", handletextpad=0.2)
    panel(ax, "a", dx=-0.2, dy=1.01)

    for row, (mp, col, letter) in enumerate((("JW", C_JW, "b"), ("TT", C_TT, "c"))):
        ax = fig.add_subplot(gs[row, 1]); tidy(ax)
        p = THEORY["profile"][mp]; s = np.arange(len(p["det"])) * p["stride"]
        for a, b, _ in p["intervals_C_B_plus_1"]:
            ax.axvspan(a, b, color=col, alpha=0.10, lw=0, zorder=0)
        ax.plot(s, np.array(p["sim"]) / 1e3, color=col, lw=1.3, label="simulated mean")
        ax.plot(s, np.array(p["det"]) / 1e3, color=INK, lw=0.9, ls=(0, (4, 2)), label="closed form")
        ax.set_xlim(0, 8100); ax.set_ylim(0, 33 if mp == "JW" else 24)
        ax.set_ylabel("stall so far\n($10^3$ cycles)", labelpad=2)
        ax.text(0.015, 0.95, "Jordan-Wigner" if mp == "JW" else "ternary tree", transform=ax.transAxes,
                ha="left", va="top", fontsize=7.2, color=col)
        if row == 0:
            ax.tick_params(labelbottom=False)
            ax.legend(loc="lower right", ncol=1, frameon=True, facecolor="white", edgecolor="none",
                      framealpha=1.0, borderpad=0.3)
        else:
            ax.set_xlabel("scheduled step $s$", labelpad=1)
        panel(ax, letter, dx=-0.16, dy=1.02)
    save(fig, "fig5_predictor")


# =====================================================================
# FIGURE 7 -- what the histogram allows, and where noise takes over
# =====================================================================
def fig_envelope():
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.4), gridspec_kw={"wspace": 0.42})
    for ax, mp, col, letter in ((axes[0], "JW", C_JW, "a"), (axes[1], "TT", C_TT, "b")):
        tidy(ax)
        rows = [e for e in THEORY["envelope"] if e["mapping"] == mp]; K = [e["K"] for e in rows]
        ax.fill_between(K, [max(e["lower"], 0) for e in rows], [e["upper_hi"] for e in rows], color=col, alpha=0.16, lw=0)
        ax.plot(K, [e["det_sorted"] for e in rows], color=col, lw=0.9, ls=(0, (4, 2)))
        ax.plot(K, [e["det_compiled"] for e in rows], color=INK, lw=1.2)
        ax.plot(K, [e["det_interleaved"] for e in rows], color=C_OK, lw=1.2)
        ax.set_xticks(range(6, 25, 6)); ax.set_xlabel("production patches $K$", labelpad=1)
        ax.set_ylim(-0.8, 30 if mp == "JW" else 24)
        ax.set_ylabel("deterministic excess (%)", labelpad=2)
        ax.text(0.04, 0.97, "Jordan-Wigner" if mp == "JW" else "ternary tree", transform=ax.transAxes,
                ha="left", va="top", fontsize=7.2, color=col)
        if mp == "JW":
            ax.legend(handles=[Patch(fc=col, alpha=0.16, lw=0), Line2D([], [], color=col, lw=0.9, ls=(0, (4, 2))),
                               Line2D([], [], color=INK, lw=1.2), Line2D([], [], color=C_OK, lw=1.2)],
                      labels=["envelope", "sorted", "compiled", "interleaved"],
                      loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=6.9)
        panel(ax, letter, dx=-0.24)
    ax = axes[2]; tidy(ax)
    for name, col, mk in (("interleaved", C_OK, "o"), ("shuffled", C_DET, "s"), ("compiled", INK, "^"), ("clustered", C_WARN, "D")):
        rows = [r for r in THEORY["order_buffer"] if r["order"] == name]
        ax.plot([r["B"] for r in rows], [r["sim"] - r["det"] for r in rows], mk + "-", color=col, mfc="white",
                mew=1.1, ms=3.6, label=name)
    ax.set_xscale("log", base=2); ax.set_xticks([4, 8, 16, 32, 64]); ax.set_xticklabels(["4", "8", "16", "32", "64"])
    ax.set_xlabel("external storage $B$", labelpad=1)
    ax.set_ylabel("simulated minus closed form (pp)", labelpad=2)
    ax.set_ylim(-0.3, 4.8)
    ax.legend(loc="upper right", fontsize=6.9)
    panel(ax, "c", dx=-0.24)
    save(fig, "fig7_envelope")


# =====================================================================
# FIGURE 9 -- relaxing the workload proxy; independently produced traces
# =====================================================================
FAMILIES = (("qft", "QFT", INK, "D"), ("qaoa", "QAOA", C_JW, "o"), ("cla_adder", "CLA adder", C_WARN, "s"),
            ("multiplier", "multiplier", C_OK, "^"), ("adder", "ripple adder", C_TT, "v"),
            ("modular_multiplier", "mod. mult.", "#6B4E9B", "P"))


def fig_validity():
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.55), gridspec_kw={"wspace": 0.42, "width_ratios": [1.0, 1.0, 1.05]})
    ax = axes[0]; tidy(ax)
    for mp, col in (("JW", C_JW), ("TT", C_TT)):
        rows = [r for r in ROBUST["budget"] if r["mapping"] == mp]
        ax.axhline(rows[0]["limit"], color=col, lw=0.8, ls=(0, (4, 2.5)), alpha=0.8)
        ax.plot([r["N"] for r in rows], [r["det"] for r in rows], "-", color=col, lw=1.0, alpha=0.55)
        ax.plot([r["N"] for r in rows], [r["excess"] for r in rows], "o", color=col, mfc="white", mew=1.2, ms=3.6)
    ax.plot([69, 69], [12.2, 27], color=INK2, lw=0.7, ls=(0, (1, 2)))
    ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 50, 100, 200]); ax.set_xticklabels(["5", "10", "20", "50", "100", "200"])
    ax.set_ylim(0, 27); ax.set_xlabel(r"states per rotation $N_{\mathrm{synth}}$", labelpad=1)
    ax.set_ylabel(r"excess over $T_{\mathrm{rate}}$ (%)", labelpad=2)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C_JW, mfc="white", mew=1.2),
                       Line2D([], [], marker="o", ls="", color=C_TT, mfc="white", mew=1.2),
                       Line2D([], [], color=C_DET, lw=1.0), Line2D([], [], color=C_DET, lw=0.8, ls=(0, (4, 2.5)))],
              labels=["JW, simulated", "TT, simulated", "closed form", r"limit $N_{\mathrm{synth}}\to\infty$"],
              loc="lower right", fontsize=6.8)
    panel(ax, "a", dx=-0.24)

    ax = axes[1]; tidy(ax, grid="both")
    ax.plot([0, 24], [0, 24], color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
    for key, lab, col, mk in FAMILIES:
        rows = [r for r in ROBUST["external"] if r["family"] == key and "excess" in r]
        ax.plot([r["det"] for r in rows], [r["excess"] for r in rows], mk, color=col, mfc="white", mew=1.0, ms=3.6,
                label=lab, zorder=3)
    ax.set_xlim(-1, 24); ax.set_ylim(-1, 33); ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_xlabel("closed-form excess (%)", labelpad=1); ax.set_ylabel("simulated mean excess (%)", labelpad=2)
    ax.legend(loc="upper left", ncol=2, fontsize=6.5, columnspacing=0.6, handletextpad=0.1)
    panel(ax, "b", dx=-0.24)

    ax = axes[2]; tidy(ax)
    rng = np.random.default_rng(3)
    labels = []
    for i, (key, lab, col, mk) in enumerate(FAMILIES):
        rows = [r for r in ROBUST["external"] if r["family"] == key]
        y = [100 * r["det"] / r["det_sorted"] for r in rows]
        ax.plot(i + rng.uniform(-0.17, 0.17, len(y)), y, mk, color=col, mfc="white", mew=1.0, ms=3.4)
        labels.append(lab)
    env = {(e["mapping"], e["K"]): e for e in THEORY["envelope"]}
    hub = [100 * env[("JW", 10)]["det_compiled"] / env[("JW", 10)]["det_sorted"],
           100 * env[("TT", 11)]["det_compiled"] / env[("TT", 11)]["det_sorted"]]
    ax.plot([len(FAMILIES) - 0.1, len(FAMILIES) + 0.1], hub, "*", color=INK, ms=6.5, mew=0)
    labels.append("Hubbard\n(this work)")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=6.6, rotation_mode="anchor")
    ax.set_ylim(-4, 104); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("closed form, % of sorted order", labelpad=2)
    panel(ax, "c", dx=-0.24)
    save(fig, "fig9_validity")


# =====================================================================
# SUPPLEMENT
# =====================================================================
def fig_regime():
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.4), gridspec_kw={"wspace": 0.3})
    ax = axes[0]; tidy(ax)
    for mp, col in (("JW", C_JW), ("TT", C_TT)):
        ps = sorted(float(k.split("_")[-1]) for k in RES if k.startswith(f"acceptance_{mp}_"))
        ax.plot(ps, [excess(f"acceptance_{mp}_{p}") for p in ps], "o", color=col, mfc="white", mew=1.2, label=f"{mp}, simulated")
        ax.plot(ps, [PRED[f"acceptance_{mp}_{p}"]["det"] for p in ps], "-", color=col, lw=1.0, alpha=0.55)
    ax.set_xlabel("acceptance probability $p$", labelpad=1)
    ax.set_ylabel(r"excess over $T_{\mathrm{rate}}$ (%)", labelpad=2)
    ax.set_ylim(-1, 27)
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], color=C_DET, lw=1.0)); l.append("closed form")
    ax.legend(h, l, loc="upper right", ncol=1)
    panel(ax, "a")

    ax = axes[1]; tidy(ax)
    for mp, col in (("JW", C_JW), ("TT", C_TT)):
        ns, ys, los, his = [], [], [], []
        for n in (1, 2, 4, 8):
            m, lo, hi = ci(f"repeated_{mp}_{n}")
            ns.append(n); ys.append(m); los.append(lo); his.append(hi)
        ax.errorbar(ns, ys, yerr=[los, his], fmt="s-", color=col, mfc="white", mew=1.2, capsize=1.6,
                    elinewidth=0.8, label=mp)
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_xlabel("concatenated trace repetitions", labelpad=1)
    ax.set_ylabel(r"excess over $T_{\mathrm{rate}}$ (%)", labelpad=2)
    ax.set_ylim(13.5, 21.5)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.5)); panel(ax, "b", dx=-0.2)
    save(fig, "figS1_regime")


def fig_laws():
    fig, ax = plt.subplots(figsize=(W2, 2.2)); tidy(ax)
    order = [("geometric", "geometric"), ("deterministic", "deterministic"),
             ("two_point", "two-point"), ("common_slowdown", "common environment")]
    x = np.arange(len(order)); wd = 0.34
    for i, (mp, col) in enumerate((("JW", C_JW), ("TT", C_TT))):
        ys = [excess(f"baseline_{mp}" if key == "geometric" else f"law_{mp}_{key}") for key, _ in order]
        ax.bar(x + (i - 0.5) * wd, ys, width=wd, color=col, lw=0, label=mp)
        for xx, yy in zip(x + (i - 0.5) * wd, ys):
            ax.text(xx, yy + 0.4, f"{yy:.1f}", ha="center", fontsize=6.9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in order], fontsize=7.5)
    ax.set_ylabel(r"excess over $T_{\mathrm{rate}}$ (%)", labelpad=2)
    ax.set_ylim(0, 31)
    ax.legend(loc="upper left", ncol=2)
    save(fig, "figS2_laws")


if __name__ == "__main__":
    print("building figures ->", FIG, "| font:", FAMILY)
    fig_model(); fig_exact(); fig_traces(); fig_capacity(); fig_predictor(); fig_order()
    fig_envelope(); fig_buffer(); fig_validity(); fig_validation(); fig_regime(); fig_laws()
    print("done")
