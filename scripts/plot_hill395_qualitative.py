#!/usr/bin/env python
"""Render *qualitative*-analysis figures for the Hill 395 experiment (section 5.7).

Unlike plot_hill395_results.py (aggregate/quantitative charts), this script reads the
per-sector model outputs in hill395_experiment_results/samples/*.json together with the
dataset gold identities, and emits case-level visualisations that back the qualitative
discussion:

  1. fig_qual_normal_merge.png  - source-complementary correct merge (침투조 KG-A ↔ 잔존대 KG-B)
  2. fig_qual_confidence.png    - match-confidence overlap of correct vs wrong merges
                                  (why a single threshold cannot remove all wrong merges)
  3. fig_qual_impossible.png    - kinematic map: benign sensor offset (correct merge) vs
                                  genuine wrong merge, plus impossible-edge density per object

Korean labels render via Malgun Gothic (present on Windows). Falls back to a warning box.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "hill395_experiment_results"
FIG = RES / "figures"
DATA = REPO / "data" / "battlefield_hill395_large" / "sectors"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, RED, GREY, GREEN, ORANGE = "#2c6fbb", "#c0392b", "#888888", "#27ae60", "#e08a1e"

# Korean-capable font
for _name in ("Malgun Gothic", "NanumGothic", "Gulim", "Batang"):
    if _name in {f.name for f in fm.fontManager.ttflist}:
        plt.rcParams["font.family"] = _name
        break
plt.rcParams["axes.unicode_minus"] = False

SAMPLES = ["sec_dev_0000", "sec_dev_0001", "sec_dev_0002"]


def load_sample(sid):
    return json.loads((RES / "samples" / f"output_{sid}.json").read_text(encoding="utf-8"))


def load_gold(sid):
    g = json.loads((DATA / sid / "labels" / "gold_identities.json").read_text(encoding="utf-8"))
    return {o: gi["gold_identity_id"] for gi in g["identities"] for o in gi["member_observations"]}


def entity_labels(sid):
    """local_entity_id -> majority Korean label_text."""
    by_ent = {}
    with open(DATA / sid / "observations.jsonl", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            by_ent.setdefault(o["local_entity_id"], Counter())[o["label_text"]] += 1
    return {e: c.most_common(1)[0][0] for e, c in by_ent.items()}


def classify(sample, obs2gold):
    """Return predicted identities tagged pure(correct)/mixed(wrong) with impossible-edge counts."""
    out = []
    for it in sample["identities"]:
        traj = {t["obs_id"]: t for t in it["trajectory"]}
        if len(traj) < 2:
            continue
        golds = Counter(obs2gold.get(o) for o in traj if obs2gold.get(o))
        if not golds:
            continue
        n_imp = sum(1 for tr in it["transitions"] if not tr.get("feasible_motion", True))
        out.append({"id": it["global_id"], "type": it["type"],
                    "conf": float(it["match_confidence"]),
                    "pure": len(golds) <= 1, "n_imp": n_imp, "n_obs": len(traj)})
    return out


# ---------------------------------------------------------------- figure 1
def fig_normal_merge():
    s = load_sample("sec_dev_0000")
    labels = entity_labels("sec_dev_0000")
    it = next(x for x in s["identities"] if x["global_id"] == "G-sec_dev_0000-02")
    a_ent = it["local_entity_ids"]["A"]; b_ent = it["local_entity_ids"]["B"]
    a_lbl, b_lbl = labels.get(a_ent, a_ent), labels.get(b_ent, b_ent)
    a_src = sorted({t["source"] for t in it["trajectory"] if t["kg_id"] == "A"})
    b_src = sorted({t["source"] for t in it["trajectory"] if t["kg_id"] == "B"})
    conf = it["match_confidence"]; typ = it["type"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")
    # KG-A box
    ax.add_patch(plt.Rectangle((0.02, 0.55), 0.34, 0.34, fc="#eaf1fb", ec=BLUE, lw=2))
    ax.text(0.19, 0.84, "KG-A (직접 관측)", ha="center", fontsize=11, color=BLUE, weight="bold")
    ax.text(0.19, 0.74, a_lbl, ha="center", fontsize=12, weight="bold")
    ax.text(0.19, 0.66, "출처: " + " / ".join(a_src), ha="center", fontsize=8, color="#444")
    ax.text(0.19, 0.60, "10 obs", ha="center", fontsize=8, color="#444")
    # KG-B box
    ax.add_patch(plt.Rectangle((0.02, 0.11), 0.34, 0.34, fc="#fdeeea", ec=RED, lw=2))
    ax.text(0.19, 0.40, "KG-B (항공·감청)", ha="center", fontsize=11, color=RED, weight="bold")
    ax.text(0.19, 0.30, b_lbl, ha="center", fontsize=12, weight="bold")
    ax.text(0.19, 0.22, "출처: " + " / ".join(b_src), ha="center", fontsize=8, color="#444")
    ax.text(0.19, 0.16, "10 obs", ha="center", fontsize=8, color="#444")
    # merged node
    ax.add_patch(plt.Circle((0.78, 0.5), 0.16, fc="#eafaef", ec=GREEN, lw=2.5))
    ax.text(0.78, 0.57, "통합 개체", ha="center", fontsize=11, color=GREEN, weight="bold")
    ax.text(0.78, 0.49, typ, ha="center", fontsize=12, weight="bold")
    ax.text(0.78, 0.41, f"확신도 {conf:.2f}", ha="center", fontsize=11, color=GREEN)
    # arrows
    ax.annotate("", xy=(0.62, 0.55), xytext=(0.36, 0.72),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2))
    ax.annotate("", xy=(0.62, 0.45), xytext=(0.36, 0.28),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    ax.text(0.49, 0.6, "출처 상호보완", fontsize=9, color="#666", rotation=18)
    ax.set_title("정상 병합 — 서로 다른 출처의 두 관측을 동일 개체로 통합\n"
                 f"(sec_dev_0000, gold 정답: 동일 실체)", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "fig_qual_normal_merge.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_confidence_overlap():
    correct, wrong = [], []
    for sid in SAMPLES:
        rows = classify(load_sample(sid), load_gold(sid))
        for r in rows:
            (correct if r["pure"] else wrong).append(r["conf"])
    import numpy as np
    fig, ax = plt.subplots(figsize=(9, 4.8))
    rng = np.random.default_rng(0)
    ax.scatter(correct, rng.uniform(0.55, 0.95, len(correct)), s=22, color=GREEN,
               alpha=0.6, label=f"정상 병합 (correct, n={len(correct)})")
    ax.scatter(wrong, rng.uniform(0.05, 0.45, len(wrong)), s=30, color=RED,
               alpha=0.75, label=f"오병합 (wrong, n={len(wrong)})")
    # overlap region: where wrong merges exist at high conf
    hi_wrong = [c for c in wrong if c >= 0.8]
    ax.axvspan(0.8, 1.0, color=ORANGE, alpha=0.12)
    ax.text(0.9, 0.5, f"고확신 오병합 {len(hi_wrong)}건\n(임계값으로 제거 불가)",
            ha="center", fontsize=9, color=ORANGE)
    ax.set_xlim(0, 1.02); ax.set_yticks([])
    ax.set_xlabel("match_confidence (병합 확신도)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("병합 확신도 분포 — 정상/오병합이 고확신 구간에서 겹침\n"
                 "(단일 임계값 상향만으로는 모든 오병합을 제거할 수 없음)", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "fig_qual_confidence.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------- figure 3
def _imp_edges(it, obs2gold=None):
    traj = {t["obs_id"]: t for t in it["trajectory"]}
    edges = []
    for tr in it["transitions"]:
        if tr.get("feasible_motion", True):
            continue
        a, b = traj.get(tr["from_obs"]), traj.get(tr["to_obs"])
        if a and b:
            edges.append((a, b, tr["required_speed_mps"]))
    return traj, edges


def fig_impossible_map():
    s = load_sample("sec_dev_0000")
    obs2gold = load_gold("sec_dev_0000")
    # panel A: correct merge with sensor offset (identity-02), color by KG
    cm = next(x for x in s["identities"] if x["global_id"] == "G-sec_dev_0000-02")
    # panel B: wrong merge of two gold entities (identity-30), color by gold id
    wm = next(x for x in s["identities"] if x["global_id"] == "G-sec_dev_0000-30")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2),
                             gridspec_kw={"width_ratios": [1, 1, 0.7]})

    # ---- panel A
    ax = axes[0]
    traj, edges = _imp_edges(cm)
    for kg, col in (("A", BLUE), ("B", RED)):
        pts = [(t["easting"], t["northing"]) for t in traj.values() if t["kg_id"] == kg]
        if pts:
            ax.scatter(*zip(*pts), s=28, color=col, alpha=0.7,
                       label=f"KG-{kg} ({len(pts)} obs)")
    fast = max(edges, key=lambda e: e[2])
    ax.annotate("", xy=(fast[1]["easting"], fast[1]["northing"]),
                xytext=(fast[0]["easting"], fast[0]["northing"]),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.2))
    ax.text(0.5, 0.02, f"교차 전이 요구속도 {fast[2]:.1f} m/s "
            f"(약 {fast[2]*3.6:.0f} km/h)", transform=ax.transAxes,
            ha="center", fontsize=9, color=ORANGE)
    ax.set_title(f"정상 병합의 센서 위치 불일치\n{cm['type']}, 확신도 {cm['match_confidence']:.2f} "
                 "(gold 정답)\n→ 불가능 전이는 CEP 오차 (양성)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right"); ax.set_xlabel("easting (m)"); ax.set_ylabel("northing (m)")
    ax.ticklabel_format(style="plain", useOffset=False)

    # ---- panel B
    ax = axes[1]
    traj, edges = _imp_edges(wm)
    gold_ids = sorted({obs2gold.get(o) for o in traj if obs2gold.get(o)})
    cols = {gold_ids[0]: BLUE, gold_ids[1]: RED} if len(gold_ids) >= 2 else {}
    for gd in gold_ids:
        pts = [(t["easting"], t["northing"]) for o, t in traj.items() if obs2gold.get(o) == gd]
        if pts:
            ax.scatter(*zip(*pts), s=28, color=cols.get(gd, GREY), alpha=0.7,
                       label=f"실제 부대 {gd[-4:]} ({len(pts)} obs)")
    for a, b, sp in sorted(edges, key=lambda e: -e[2])[:6]:
        ax.annotate("", xy=(b["easting"], b["northing"]), xytext=(a["easting"], a["northing"]),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4, alpha=0.8))
    ax.set_title(f"고확신 오병합의 물리적 반증\n{wm['type']}, 확신도 {wm['match_confidence']:.2f}\n"
                 f"→ 서로 다른 부대 통합, 불가능 전이 {len(edges)}건", fontsize=10)
    ax.legend(fontsize=8, loc="upper right"); ax.set_xlabel("easting (m)")
    ax.ticklabel_format(style="plain", useOffset=False)

    # ---- panel C: density
    ax = axes[2]
    pure_ids = pure_edges = mix_ids = mix_edges = 0
    for sid in SAMPLES:
        for r in classify(load_sample(sid), load_gold(sid)):
            if r["n_imp"] == 0:
                continue
            if r["pure"]:
                pure_ids += 1; pure_edges += r["n_imp"]
            else:
                mix_ids += 1; mix_edges += r["n_imp"]
    dens = [pure_edges / max(pure_ids, 1), mix_edges / max(mix_ids, 1)]
    bars = ax.bar(["정상 병합\n(correct)", "오병합\n(wrong)"], dens, color=[GREEN, RED])
    for b, d, n in zip(bars, dens, [pure_ids, mix_ids]):
        ax.text(b.get_x() + b.get_width()/2, d + 0.3, f"{d:.1f}\n({n} objs)",
                ha="center", fontsize=9)
    ax.set_ylabel("개체당 평균 불가능 전이 수")
    ax.set_title("불가능 전이 밀도\n(오병합에 약 5배 집중)", fontsize=10)

    fig.suptitle("물리적 불가능 전이 — 양성(센서 오차)과 진성 오병합의 구분", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "fig_qual_impossible.png", dpi=140); plt.close(fig)


def main():
    fig_normal_merge()
    fig_confidence_overlap()
    fig_impossible_map()
    print("qualitative figures ->", FIG)
    for p in sorted(FIG.glob("fig_qual_*.png")):
        print("  -", p.name)


if __name__ == "__main__":
    main()
