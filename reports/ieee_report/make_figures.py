"""Generate manuscript figures and a values file from committed Phase 1 artifacts.

Every plotted number is read from results/metrics/ (and, for split sizes, from the
regenerated data/processed/scaffold_split_indices.json). Nothing is hard-coded except
labels and colours. Run from the repository root:

    python reports/ieee_report/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

METRICS = Path("results/metrics")
OUT = Path("reports/ieee_report/figures")
OUT.mkdir(parents=True, exist_ok=True)

ENDPOINTS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]
MODELS = [("logistic_regression", "LR"), ("random_forest", "RF"), ("xgboost", "XGBoost")]
FEATS = [("morgan", "Morgan (2048)"), ("descriptors", "Descriptors (9)"), ("combined", "Combined (2057)")]
# Okabe-Ito colour-blind-safe palette
FEAT_COLORS = {"morgan": "#0072B2", "descriptors": "#E69F00", "combined": "#009E73"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


summary = json.loads((METRICS / "model_benchmark_summary.json").read_text())
diag = json.loads((METRICS / "endpoint_label_diagnostics.json").read_text())["overall"]

# ---------------------------------------------------------------- values file
values = {"configs": {}, "rf_combined_endpoints": {}, "label_stats": {}}
for mk, _ in MODELS:
    for fk, _ in FEATS:
        m = summary[f"{mk}_{fk}"]["macro_metrics"]
        values["configs"][f"{mk}_{fk}"] = {
            "roc_mean": m["roc_auc"]["mean"], "roc_sd": m["roc_auc"]["sd"], "roc_sem": m["roc_auc"]["sem"],
            "ap_mean": m["pr_auc"]["mean"], "ap_sd": m["pr_auc"]["sd"], "ap_sem": m["pr_auc"]["sem"],
            "f1_mean": m["f1"]["mean"], "bacc_mean": m["balanced_accuracy"]["mean"],
            "endpoint_roc": [summary[f"{mk}_{fk}"]["endpoints"][e]["test"]["roc_auc"] for e in ENDPOINTS],
            "endpoint_ap": [summary[f"{mk}_{fk}"]["endpoints"][e]["test"]["pr_auc"] for e in ENDPOINTS],
        }
for e in ENDPOINTS:
    ep = summary["random_forest_combined"]["endpoints"][e]
    cm = ep["test"]["confusion_matrix"]
    pos, neg = cm["tp"] + cm["fn"], cm["tn"] + cm["fp"]
    values["rf_combined_endpoints"][e] = {
        "threshold": ep["val_threshold"], "roc": ep["test"]["roc_auc"], "ap": ep["test"]["pr_auc"],
        "f1": ep["test"]["f1"], "precision": ep["test"]["precision"], "recall": ep["test"]["recall"],
        "test_pos": pos, "test_obs": pos + neg, "test_prev": pos / (pos + neg),
        "train_pos": ep["train_positives"], "train_obs": ep["train_samples"],
    }
    d = diag[e]
    values["label_stats"][e] = {k: d[k] for k in ("positive_count", "negative_count", "missing_samples",
                                                   "missing_percentage", "prevalence")}
split_path = Path("data/processed/scaffold_split_indices.json")
if split_path.exists():
    s = json.loads(split_path.read_text())
    s = s.get("indices", s)
    values["split_sizes"] = {k: len(s[k]) for k in ("train", "val", "test")}
(OUT / "figure_values.json").write_text(json.dumps(values, indent=2))

# ------------------------------------------------- Fig: model x feature bars
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3))
x = np.arange(len(MODELS))
w = 0.26
for ax, metric, label in ((axes[0], "roc", "Test macro ROC-AUC"), (axes[1], "ap", "Test macro AP (PR-AUC)")):
    for i, (fk, flabel) in enumerate(FEATS):
        means = [values["configs"][f"{mk}_{fk}"][f"{metric}_mean"] for mk, _ in MODELS]
        sems = [values["configs"][f"{mk}_{fk}"][f"{metric}_sem"] for mk, _ in MODELS]
        bars = ax.bar(x + (i - 1) * w, means, w, yerr=sems, capsize=2, color=FEAT_COLORS[fk],
                      label=flabel, error_kw={"elinewidth": 0.7, "capthick": 0.7})
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, v + max(sems) + 0.008, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=5.5, rotation=90)
    ax.set_xticks(x, [m for _, m in MODELS])
    ax.set_ylabel(label)
    if metric == "roc":
        ax.set_ylim(0.5, 0.85)
        ax.axhline(0.5, color="0.5", lw=0.6, ls=":")
    else:
        ax.set_ylim(0.0, 0.55)
    ax.grid(axis="y", color="0.9", lw=0.5)
    ax.set_axisbelow(True)
axes[0].legend(loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.0, 1.18))
save(fig, "fig_model_comparison")

# ------------------------------------------- Fig: RF-combined per endpoint
rf = values["rf_combined_endpoints"]
order = sorted(ENDPOINTS, key=lambda e: rf[e]["roc"])
fig, ax = plt.subplots(figsize=(3.5, 2.9))
y = np.arange(len(order))
ax.barh(y + 0.2, [rf[e]["roc"] for e in order], 0.38, color="#0072B2", label="ROC-AUC")
ax.barh(y - 0.2, [rf[e]["ap"] for e in order], 0.38, color="#009E73", label="AP")
ax.scatter([rf[e]["test_prev"] for e in order], y - 0.2, marker="|", s=60, color="black", lw=1.2,
           label="AP baseline (test prevalence)", zorder=3)
ax.axvline(0.5, color="0.5", lw=0.6, ls=":")
ax.set_yticks(y, order)
ax.set_xlim(0, 1)
ax.set_xlabel("Test score (RF, combined features)")
ax.grid(axis="x", color="0.9", lw=0.5)
ax.set_axisbelow(True)
ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, frameon=False, fontsize=6, handlelength=1.2, columnspacing=0.8)
save(fig, "fig_endpoint_rf_combined")

# ---------------------------------------- Fig: label prevalence/missingness
ls = values["label_stats"]
fig, ax1 = plt.subplots(figsize=(3.5, 2.2))
xe = np.arange(len(ENDPOINTS))
ax1.bar(xe - 0.2, [100 * ls[e]["prevalence"] for e in ENDPOINTS], 0.4, color="#D55E00", label="Positive prevalence (% of observed labels)")
ax1.bar(xe + 0.2, [ls[e]["missing_percentage"] for e in ENDPOINTS], 0.4, color="0.6", label="Missing labels (% of 7,823 molecules)")
ax1.set_xticks(xe, ENDPOINTS, rotation=60, ha="right")
ax1.set_ylabel("Percent")
ax1.grid(axis="y", color="0.9", lw=0.5)
ax1.set_axisbelow(True)
ax1.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1)
save(fig, "fig_label_statistics")

# ------------------------------------------------------ Fig: pipeline diagram
fig, ax = plt.subplots(figsize=(7.16, 1.05))
ax.set_xlim(-1.5, 101.5)
ax.set_ylim(0, 15)
ax.axis("off")
stages = [
    ("Tox21 CSV\n7,831 rows\nSHA-256 checked", "#E8EEF6"),
    ("RDKit sanitise\n& canonicalise\n7,823 kept, 8 excl.", "#E8EEF6"),
    ("Features\nMorgan r=2, 2048 bits\n+ 9 descriptors\n= 2057 combined", "#E6F3EE"),
    ("Bemis–Murcko\nscaffold split\n6,258 / 782 / 783", "#FDF1DE"),
    ("Train LR, RF, XGB\nper endpoint\nNaN labels masked\nclass weighting", "#E6F3EE"),
    ("Threshold\n= argmax val F1\n(validation only)", "#FDF1DE"),
    ("Test evaluation\nROC-AUC, AP, F1\nmacro over 12", "#E8EEF6"),
]
bw, gap = 12.9, 1.6
for i, (txt, col) in enumerate(stages):
    x0 = i * (bw + gap)
    ax.add_patch(FancyBboxPatch((x0, 1), bw, 13, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc=col, ec="0.35", lw=0.6))
    ax.text(x0 + bw / 2, 7.5, txt, ha="center", va="center", fontsize=5.8, linespacing=1.2)
    if i < len(stages) - 1:
        ax.add_patch(FancyArrowPatch((x0 + bw + 0.35, 7.5), (x0 + bw + gap - 0.35, 7.5),
                                     arrowstyle="-|>", mutation_scale=7, lw=0.7, color="0.25"))
save(fig, "fig_pipeline")

print("Wrote figures and figure_values.json to", OUT)
