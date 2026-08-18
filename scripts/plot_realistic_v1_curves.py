"""dev-F1 progress curves per ablation variant, parsed from the training logs.

IMPORTANT — what this can and cannot show (read before interpreting):

engine_large.train_large evaluates dev every max(1, log_every//2) = 5 epochs on
dev_eval_sectors=6 sectors, but only PRINTS every log_every = 10 epochs, and what
it prints is `best_dev_f1` -- the running MAXIMUM, not that epoch's evaluation:

    if epoch % 5 == 0 or epoch == last:  agg = evaluate(...); best_f1 = max(best_f1, agg.f1)
    if epoch % 10 == 0 or epoch == 1:    print(..., best_dev_f1=best_f1, ...)

So from the stored logs we can only recover a **best-so-far envelope at epochs
10/20/30/40**, which is monotone non-decreasing by construction. The per-epoch
dev-F1 at the 5-epoch marks was never written to disk and cannot be recovered
without retraining. Epoch 1 prints -1.0 (sentinel: no evaluation has run yet) and
is dropped.

Two further caveats:
  * these values come from a 6-sector dev subsample (the checkpoint-selection
    proxy), NOT the 12-sector dev aggregate reported in metrics.json;
  * a monotone envelope hides within-run dips, so "no crossing" here is weaker
    evidence of rank stability than a true learning curve would be.

    python scripts/plot_realistic_v1_curves.py                  # pilot (all seeds found)
    python scripts/plot_realistic_v1_curves.py --seeds 19521006 --out <path>
"""
from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "runs" / "realistic_v1" / "ablation"
MODELS = ["m3_full", "no_motion", "no_time", "no_state", "no_rel", "no_src"]
LINE = re.compile(r"^epoch\s+(\d+)/(\d+)\s+loss=([\d.]+)\s+best_dev_f1=(-?[\d.]+)")


def parse_log(path: Path):
    """-> {epoch: (loss, best_dev_f1)}, dropping the epoch-1 '-1.0' sentinel."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE.match(line)
        if m:
            ep, _, loss, f1 = int(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))
            if f1 >= 0.0:                      # -1.0 => no dev eval had run yet
                out[ep] = (loss, f1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="seeds to include (default: every seed found in the run dir)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title-suffix", default="")
    args = ap.parse_args()

    seeds = args.seeds
    if not seeds:
        seeds = sorted({int(m.group(1)) for p in RUN.glob("*_seed*.log")
                        if (m := re.search(r"_seed(\d+)\.log$", p.name))})
    if not seeds:
        raise SystemExit(f"no training logs found under {RUN}")

    # model -> epoch -> [f1 across seeds]
    data, losses = {}, {}
    for mdl in MODELS:
        per_ep, per_loss = {}, {}
        for s in seeds:
            for ep, (loss, f1) in parse_log(RUN / f"{mdl}_seed{s}.log").items():
                per_ep.setdefault(ep, []).append(f1)
                per_loss.setdefault(ep, []).append(loss)
        if per_ep:
            data[mdl], losses[mdl] = per_ep, per_loss
    if not data:
        raise SystemExit("no parsable epoch lines found")

    n = max(len(v) for m in data.values() for v in m.values())
    out = Path(args.out) if args.out else (
        REPO / "results" / "realistic_v1_ablation" /
        (f"curves_dev_f1_pilot.png" if n == 1 else f"curves_dev_f1_n{n}.png"))
    out.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    colors = {"m3_full": "#2c7fb8", "no_motion": "#d95f02", "no_time": "#7570b3",
              "no_state": "#e7298a", "no_rel": "#66a61e", "no_src": "#a6761d"}

    ax = axes[0]
    for mdl, per_ep in data.items():
        eps = sorted(per_ep)
        mus = [statistics.mean(per_ep[e]) for e in eps]
        ax.plot(eps, mus, marker="o", lw=2 if mdl == "m3_full" else 1.5,
                ls="-" if mdl == "m3_full" else "--", color=colors[mdl], label=mdl)
        if n > 1:
            sds = [statistics.stdev(per_ep[e]) if len(per_ep[e]) > 1 else 0.0 for e in eps]
            ax.fill_between(eps, [m - s for m, s in zip(mus, sds)],
                            [m + s for m, s in zip(mus, sds)], color=colors[mdl], alpha=0.12)
    ax.set_xlabel("epoch"); ax.set_ylabel("best-so-far dev F1 (6-sector proxy)")
    ax.set_title("dev F1 — best-so-far envelope (NOT a per-epoch curve)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    for mdl, per_ep in losses.items():
        eps = sorted(per_ep)
        ax.plot(eps, [statistics.mean(per_ep[e]) for e in eps], marker="o",
                lw=2 if mdl == "m3_full" else 1.5,
                ls="-" if mdl == "m3_full" else "--", color=colors[mdl], label=mdl)
    ax.set_xlabel("epoch"); ax.set_ylabel("train total loss")
    ax.set_title("training loss")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle(f"realistic-v1 ablation — training progress (n={n} seed{'s' if n > 1 else ''}"
                 f"{', ' + ', '.join(str(s) for s in seeds) if n <= 5 else ''})"
                 f"{args.title_suffix}\n"
                 "values are the running MAXIMUM printed every 10 epochs on a 6-sector dev "
                 "subsample — monotone by construction; per-5-epoch values were never logged",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=140)

    # ---------- reading ----------
    eps_common = sorted(set.intersection(*[set(v) for v in data.values()]))
    order = {e: [m for m in sorted(data, key=lambda m: -statistics.mean(data[m][e]))]
             for e in eps_common}
    crossings = sum(1 for a, b in zip(eps_common, eps_common[1:]) if order[a] != order[b])
    last, prev = eps_common[-1], eps_common[-2] if len(eps_common) > 1 else eps_common[-1]
    slopes = {m: (statistics.mean(data[m][last]) - statistics.mean(data[m][prev]))
              for m in data}
    smin, smax = min(slopes.values()), max(slopes.values())

    print(f"WROTE: {out}")
    print(f"\nrank order by epoch (best -> worst):")
    for e in eps_common:
        print(f"  ep{e:3d}: " + " > ".join(order[e]))
    print(f"\nrank changes between logged points: {crossings}/{len(eps_common)-1}")
    print(f"final-segment slope (ep{prev}->ep{last}) per variant:")
    for m, s in sorted(slopes.items(), key=lambda kv: -kv[1]):
        print(f"   {m:11} {s:+.4f}")
    verdict = ("순위 안정 (곡선 평행)" if crossings == 0 and (smax - smin) < 0.02 else
               "이 순위는 40에폭 예산 의존적일 수 있음 (곡선 교차 또는 기울기 상이)")
    print(f"\n판독: {verdict}")
    print(f"  근거: 로그된 지점 간 순위 변동 {crossings}회, 최종구간 기울기 범위 "
          f"{smin:+.4f}~{smax:+.4f} (폭 {smax - smin:.4f}).")
    print("  주의: 위 값은 러닝 최댓값 포락선이라 단조 증가가 강제됨 -> 실제 곡선보다 "
          "교차가 과소평가된다. 순위 안정으로 보여도 근거는 약하다.")


if __name__ == "__main__":
    main()
