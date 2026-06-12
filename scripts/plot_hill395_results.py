#!/usr/bin/env python
"""Render quantitative-evaluation figures for the Hill 395 experiment.

Reads the artifacts written by extract_hill395_results.py and emits PNG charts to
hill395_experiment_results/figures/. Labels are kept ASCII/English so they render
without a Korean font installed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "hill395_experiment_results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, RED, GREY = "#2c6fbb", "#c0392b", "#888888"


def _load_csv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fig_training_curve():
    rows = _load_csv(RES / "training_curve.csv")
    ep = [int(r["epoch"]) for r in rows]
    loss = [float(r["loss"]) for r in rows]
    f1 = [float(r["best_dev_f1"]) for r in rows]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(ep, loss, color=RED, marker="o", ms=3, label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("total loss", color=RED)
    ax1.tick_params(axis="y", labelcolor=RED)
    ax2 = ax1.twinx()
    f1c = [v if v >= 0 else None for v in f1]
    ax2.plot(ep, f1c, color=BLUE, marker="s", ms=3, label="best dev F1")
    ax2.set_ylabel("best dev F1", color=BLUE); ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", labelcolor=BLUE)
    plt.title("Hill 395 - training loss & best dev F1 (40 epochs)")
    fig.tight_layout(); fig.savefig(FIG / "training_curve.png", dpi=140); plt.close(fig)


def fig_dev_test_metrics():
    s = json.loads((RES / "metrics_summary.json").read_text(encoding="utf-8"))
    keys = ["precision", "recall", "f1", "wrong_merge_rate", "fragmentation_rate",
            "trajectory_consistency_rate", "impossible_transition_rate",
            "dangling_precision", "dangling_recall"]
    short = ["prec", "recall", "F1", "wrong\nmerge", "frag", "traj\nconsist",
             "imposs\ntrans", "dang\nprec", "dang\nrecall"]
    dev = [s["dev"][k] for k in keys]; test = [s["test"][k] for k in keys]
    x = range(len(keys)); w = 0.4
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar([i - w/2 for i in x], dev, w, label="dev", color=BLUE)
    ax.bar([i + w/2 for i in x], test, w, label="test", color=RED)
    for i in x:
        ax.text(i - w/2, dev[i] + 0.01, f"{dev[i]:.2f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w/2, test[i] + 0.01, f"{test[i]:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x)); ax.set_xticklabels(short, fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("score / rate"); ax.legend()
    ax.set_title("Hill 395 - dev/test aggregate metrics")
    fig.tight_layout(); fig.savefig(FIG / "dev_test_metrics.png", dpi=140); plt.close(fig)


def fig_baseline_compare():
    s = json.loads((RES / "metrics_summary.json").read_text(encoding="utf-8"))
    keys = ["precision", "recall", "f1"]
    hill = [s["test"][k] for k in keys]
    base = [s["baseline_large"][k] for k in keys]
    x = range(len(keys)); w = 0.4
    fig, ax = plt.subplots(figsize=(6, 4.3))
    ax.bar([i - w/2 for i in x], base, w, label="baseline large suite", color=GREY)
    ax.bar([i + w/2 for i in x], hill, w, label="Hill 395 (test)", color=RED)
    for i in x:
        ax.text(i - w/2, base[i] + 0.01, f"{base[i]:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, hill[i] + 0.01, f"{hill[i]:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(["precision", "recall", "F1"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("score"); ax.legend()
    ax.set_title("Hill 395 vs original large suite (P/R/F1)")
    fig.tight_layout(); fig.savefig(FIG / "baseline_compare.png", dpi=140); plt.close(fig)


def fig_per_sector():
    rows = _load_csv(RES / "per_sector_metrics.csv")
    rows.sort(key=lambda r: (r["split"], r["scenario_id"]))
    names = [r["scenario_id"].replace("sec_", "") for r in rows]
    f1 = [float(r["f1"]) for r in rows]
    prec = [float(r["precision"]) for r in rows]
    rec = [float(r["recall"]) for r in rows]
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(x, f1, color=BLUE, label="F1")
    ax.plot(x, prec, color=RED, marker="o", ms=3, lw=1, label="precision")
    ax.plot(x, rec, color="#27ae60", marker="^", ms=3, lw=1, label="recall")
    ndev = sum(1 for r in rows if r["split"] == "dev")
    ax.axvline(ndev - 0.5, color="k", ls="--", lw=1)
    ax.text(ndev/2, 1.02, "dev", ha="center", fontsize=9)
    ax.text(ndev + (len(rows)-ndev)/2, 1.02, "test", ha="center", fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_ylim(0, 1.08); ax.set_ylabel("score"); ax.legend(loc="lower right")
    ax.set_title("Hill 395 - per-sector precision / recall / F1")
    fig.tight_layout(); fig.savefig(FIG / "per_sector_metrics.png", dpi=140); plt.close(fig)


def fig_distributions():
    st = json.loads((RES / "dataset_stats.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, key, title, color in (
        (axes[0], "type_distribution", "object type", BLUE),
        (axes[1], "state_distribution", "state", "#27ae60"),
        (axes[2], "source_distribution", "source (KG-A/B)", RED)):
        d = st[key]
        labels = list(d.keys()); vals = list(d.values())
        ax.barh(range(len(labels)), vals, color=color)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis(); ax.set_title(f"{title} distribution"); ax.set_xlabel("observations")
    fig.suptitle("Hill 395 dataset - observation distributions (51,310 obs)")
    fig.tight_layout(); fig.savefig(FIG / "dataset_distributions.png", dpi=140); plt.close(fig)


def fig_confusion():
    rows = _load_csv(RES / "type_confusion.csv")
    rows = rows[:12]
    labels = [r["type_pair"] for r in rows]
    vals = [int(r["wrong_merge_count"]) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(range(len(labels)), vals, color=RED)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("wrong-merge count")
    ax.set_title("Hill 395 - most-confused type pairs (wrong merges)")
    fig.tight_layout(); fig.savefig(FIG / "type_confusion.png", dpi=140); plt.close(fig)


def main():
    fig_training_curve()
    fig_dev_test_metrics()
    fig_baseline_compare()
    fig_per_sector()
    fig_distributions()
    fig_confusion()
    print("figures ->", FIG)
    for p in sorted(FIG.glob("*.png")):
        print("  -", p.name)


if __name__ == "__main__":
    main()
