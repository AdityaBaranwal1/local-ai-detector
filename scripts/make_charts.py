"""README charts (light + dark PNGs) from cached evaluation results. Reads aggregates only, never prints texts.

  .venv/bin/python scripts/make_charts.py --models v0:"tropa-mini (start)" k3:"k3 (narrow data)" [v4:"v4 ..."]
Writes docs/img/{headtohead,bias_domain,ai_styles,speed}-{light,dark}.png.
Palette: the dataviz skill's validated categorical slots (blue, orange, aqua, yellow), light and dark steps;
every bar carries its value label (relief for the light-mode contrast warning on aqua/yellow).
"""
import argparse
import json
import pathlib
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.bias import domain_of, groups, style_of  # noqa: E402
from eval.suite import RUNS, load_sets, rule_fn  # noqa: E402

OUT = ROOT / "docs/img"
THEME = {
    "light": {"bg": "#fcfcfb", "fg": "#0b0b0b", "fg2": "#52514e", "grid": "#e4e3df",
              "series": ["#2a78d6", "#1baf7a", "#eda100", "#6250d6"], "ref": "#eb6834"},
    "dark": {"bg": "#1a1a19", "fg": "#f0efec", "fg2": "#c3c2b7", "grid": "#33322f",
             "series": ["#3987e5", "#199e70", "#c98500", "#9085e9"], "ref": "#d95926"},
}


def style(ax, t):
    ax.set_facecolor(t["bg"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["fg2"], labelsize=9, length=0)
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def hbars(ax, cats, series, t, fmt="{:.0%}", xmax=1.0):
    """series: [(label, color, [values per cat])]; grouped horizontal bars, 2px gaps, value labels."""
    n = len(series)
    h = 0.8 / n
    for i, (lab, col, vals) in enumerate(series):
        ys = [c + i * h - 0.4 + h / 2 for c in range(len(cats))]   # series 1 on top, as in the legend
        bars = ax.barh(ys, [v if v is not None else 0 for v in vals], height=h * 0.86, color=col, label=lab,
                       edgecolor=t["bg"], linewidth=1)
        for y, v in zip(ys, vals):
            if v is not None:
                ax.text(v + xmax * 0.01, y, fmt.format(v), va="center", fontsize=8, color=t["fg2"])
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, color=t["fg"], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.12)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def save(fig, name, mode):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}-{mode}.png", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def rate(rs, f, probs):
    return sum(f(probs[r["k"]]) >= .5 for r in rs) / len(rs) if rs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="run:label (eval/runs/<run>.json)")
    ap.add_argument("--rule", default="hybrid")
    a = ap.parse_args()
    models = [(m.split(":", 1)[0], m.split(":", 1)[1] if ":" in m else m) for m in a.models]
    runs = {k: json.loads((RUNS / f"{k}.json").read_text()) for k, _ in models}
    f = rule_fn(a.rule)
    rows = [r for r in load_sets() if all(r["k"] in runs[k] for k, _ in models)]

    # 1) head-to-head on the texts ZeroGPT has judged
    zz = [r for r in rows if r["zerogpt"] is not None]
    hum_groups = [("Native-English human", lambda r: r["g"] == "H native"), ("Non-native human", lambda r: r["g"] == "H non-native")]
    ai_groups = [("Raw AI", lambda r: r["g"] == "AI raw"), ("Humanized AI", lambda r: r["g"] == "AI humanized"),
                 ("AI-polished human", lambda r: r["g"] == "AI-polished human")]
    for mode, t in THEME.items():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
        fig.set_facecolor(t["bg"])
        for ax, grp, title in ((a1, hum_groups, "Human texts wrongly flagged as AI  (lower is better)"),
                               (a2, ai_groups, "AI texts caught  (higher is better)")):
            style(ax, t)
            series = []
            for i, (k, lab) in enumerate(models):
                series.append((lab, t["series"][i], [rate([r for r in zz if g(r)], f, runs[k]) for _, g in grp]))
            series.append(("ZeroGPT", t["ref"], [(sum(r["zerogpt"] >= 50 for r in zz if g(r)) / max(1, sum(1 for r in zz if g(r)))) for _, g in grp]))
            top = max([v for _, _, vs in series for v in vs if v is not None] + [0.05])
            hbars(ax, [f"{n}  (n={sum(1 for r in zz if g(r))})" for n, g in grp], series, t,
                  xmax=1.0 if ax is a2 else min(1.0, max(0.25, top * 1.15)))
            ax.set_title(title, color=t["fg"], fontsize=10.5, loc="left", pad=10)
        h, l = a2.get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, labelcolor=t["fg"], fontsize=9.5, bbox_to_anchor=(0.5, -0.06))
        fig.suptitle(f"Head-to-head with ZeroGPT on the {len(zz)} test texts it has judged", color=t["fg"], fontsize=12.5, x=0.01, ha="left", y=1.04)
        save(fig, "headtohead", mode)

    # 2) bias: human false-positive rate by domain (all held-out human texts)
    hum = [r for r in rows if r["label"] == 0]
    by = defaultdict(list)
    for r in hum:
        by[groups(r)["domain"]].append(r)
    if any(r["g"] == "H non-native" for r in hum):
        by["non-native writers"] = [r for r in hum if r["g"] == "H non-native"]
    cats = [d for d, rs in sorted(by.items(), key=lambda kv: -len(kv[1])) if len(rs) >= 20]
    for mode, t in THEME.items():
        fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(cats) * max(1, len(models)) * 0.75 + 1.2))
        fig.set_facecolor(t["bg"])
        style(ax, t)
        series = [(lab, t["series"][i], [rate(by[c], f, runs[k]) for c in cats]) for i, (k, lab) in enumerate(models)]
        hbars(ax, [f"{c}  (n={len(by[c])})" for c in cats], series, t, xmax=max(0.2, max(v for _, _, vs in series for v in vs if v is not None)))
        ax.set_title("Bias check: human texts wrongly flagged as AI, by kind of writing  (lower is better)", color=t["fg"], fontsize=11, loc="left", pad=10)
        ax.legend(loc="lower right", frameon=False, labelcolor=t["fg"], fontsize=9)
        save(fig, "bias_domain", mode)

    # 3) AI side: catch rate by style
    ai = [r for r in rows if r["label"] == 1]
    st = defaultdict(list)
    for r in ai:
        st[style_of(r)].append(r)
    cats = [c for c in ("raw", "humanized / paraphrased", "AI-polished human") if st.get(c)]
    for mode, t in THEME.items():
        fig, ax = plt.subplots(figsize=(8.5, 2.6))
        fig.set_facecolor(t["bg"])
        style(ax, t)
        series = [(lab, t["series"][i], [rate(st[c], f, runs[k]) for c in cats]) for i, (k, lab) in enumerate(models)]
        hbars(ax, [f"{c}  (n={len(st[c])})" for c in cats], series, t)
        ax.set_title("AI texts caught, by how the AI text was made  (higher is better)", color=t["fg"], fontsize=11, loc="left", pad=10)
        ax.legend(loc="lower right", frameon=False, labelcolor=t["fg"], fontsize=9)
        save(fig, "ai_styles", mode)

    # 4) speed on the GTX 1660 Ti (eval/BENCH.txt)
    bench = ROOT / "eval/BENCH.txt"
    pts = []
    for line in bench.read_text().splitlines():
        p = line.split()
        if len(p) >= 4 and p[0].isdigit() and p[1].isdigit():
            pts.append((int(p[0]), float(p[2]), float(p[3])))
    if pts:
        for mode, t in THEME.items():
            fig, ax = plt.subplots(figsize=(7, 3.2))
            fig.set_facecolor(t["bg"])
            style(ax, t)
            ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
            ax.xaxis.grid(False)
            xs = [p[0] for p in pts]
            for lab, idx, col in (("GPU (GTX 1660 Ti)", 1, t["series"][0]), ("CPU fallback (Ryzen 7 4800H)", 2, t["ref"])):
                ys = [p[idx] for p in pts]
                ax.plot(xs, ys, color=col, linewidth=2, marker="o", markersize=5, label=lab)
                ax.text(xs[-1] * 1.02, ys[-1], f"{ys[-1]:.0f} s", color=t["fg2"], fontsize=8.5, va="center")
            ax.set_xlabel("words per check", color=t["fg2"], fontsize=9)
            ax.set_ylabel("seconds", color=t["fg2"], fontsize=9)
            ax.set_title("Seconds per check on a 6 GB laptop GPU (whole text + per-sentence highlights)", color=t["fg"], fontsize=10.5, loc="left", pad=10)
            ax.legend(frameon=False, labelcolor=t["fg"], fontsize=9, loc="upper left")
            save(fig, "speed", mode)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
