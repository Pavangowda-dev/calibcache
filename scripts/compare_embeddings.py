# Compares two embedding models on the CALIBRATION half only:
# 1) how well each separates same-topic pairs from sibling-topic pairs (AUC)
# 2) the best hit rate at each false-hit level
# 3) a chart saved in results/compare_embeddings.png
import csv
import os
import sys

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

try:
    import matplotlib
    matplotlib.use("Agg")  # draw into a file, no window
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit('matplotlib is missing. Run:  pip install -e ".[eval]"')

from calibcache.dataset import load_queries
from calibcache.embed_queries import embed_queries
from calibcache.simulate import auc, pair_groups, sweep
from calibcache.usage_log import UsageLog

sys.stdout.reconfigure(encoding="utf-8")

MODELS = ["text-embedding-3-small", "text-embedding-3-large"]
THRESHOLDS = [round(0.40 + 0.01 * k, 2) for k in range(60)]   # 0.40 ... 0.99
NUM_ORDERS = 20
TARGETS = [0.10, 0.05, 0.02, 0.0]   # false-hit levels to compare
MIN_HITS = 10                       # ignore thresholds with fewer hits than this
CHART_FILE = "results/compare_embeddings.png"


def best_hit_rate(results, target):
    """Highest hit rate among thresholds whose false-hit rate is at most the target."""
    best = None
    for r in results:
        if r["hits"] >= MIN_HITS and r["false_hit_rate"] <= target:
            if best is None or r["hit_rate"] > best["hit_rate"]:
                best = r
    return best


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.startswith("sk-"):
        raise SystemExit("OPENAI_API_KEY is missing or wrong in .env")

    client = OpenAI(api_key=api_key)
    log = UsageLog()
    spent_before = log.total_cost()

    rows = [r for r in load_queries("data/queries_clean.jsonl") if r["split"] == "calibration"]
    n = len(rows)
    print(f"Calibration split only: {n} queries (the test split is not touched)\n")

    topic_names = sorted({r["intent_id"] for r in rows})
    family_names = sorted({r["family"] for r in rows})
    labels = np.array([topic_names.index(r["intent_id"]) for r in rows])
    family_labels = np.array([family_names.index(r["family"]) for r in rows])
    is_near_miss = np.array([r["style"] == "minimal_edit" for r in rows])
    orders = [np.random.default_rng(seed).permutation(n) for seed in range(NUM_ORDERS)]

    all_results = {}
    all_pairs = {}

    for model in MODELS:
        print(f"===== {model} =====")
        vectors = embed_queries(client, log, rows, f"results/embeddings_calibration_{model}.npz",
                                model=model)
        sim = vectors @ vectors.T

        # --- how well are the groups separated? ---
        same, sibling, other = pair_groups(sim, labels, family_labels)
        all_pairs[model] = (same, sibling)
        print("Similarity of pairs of queries (5th, 25th, 50th, 75th, 95th percentile):")
        for name, scores in [("same topic", same), ("sibling topics", sibling),
                             ("other families", other)]:
            q = np.percentile(scores, [5, 25, 50, 75, 95])
            print(f"  {name:<15} " + "  ".join(f"{v:.3f}" for v in q) + f"   ({len(scores)} pairs)")
        print(f"AUC same-topic vs sibling-topic: {auc(same, sibling):.3f}  "
              "(1.0 = perfect, 0.5 = coin flip)")

        # --- the sweep ---
        results = sweep(sim, labels, is_near_miss, THRESHOLDS, orders)
        all_results[model] = results

        print(f"\nBest hit rate at each false-hit level (needs at least {MIN_HITS} hits):")
        for target in TARGETS:
            r = best_hit_rate(results, target)
            label = "0 wrong hits" if target == 0 else f"false-hit <= {100 * target:.0f}%"
            if r is None:
                print(f"  {label:<18}: not reached")
            else:
                print(f"  {label:<18}: hit rate {100 * r['hit_rate']:.1f}% at threshold "
                      f"{r['threshold']:.2f} ({r['hits']} hits, {r['wrong']} wrong, "
                      f"95% upper {100 * r['ci_high']:.1f}%)")

        # --- save every threshold ---
        path = f"results/sweep_{model}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nAll thresholds saved in {path}\n")

    # ---------- the chart ----------
    styles = {
        MODELS[0]: {"color": "black", "linestyle": "-", "marker": "o"},
        MODELS[1]: {"color": "#777777", "linestyle": "--", "marker": "s"},
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    for model in MODELS:
        usable = [r for r in all_results[model] if r["hits"] >= MIN_HITS]
        ax.plot([100 * r["hit_rate"] for r in usable],
                [100 * r["false_hit_rate"] for r in usable],
                label=model, markersize=3, linewidth=1.4, **styles[model])
    ax.axhline(5, color="#bbbbbb", linewidth=1)
    ax.text(0.5, 5.4, "5%", color="#888888", fontsize=8)
    ax.set_xlabel("Hit rate (% of queries answered from the cache)")
    ax.set_ylabel("False-hit rate (% of hits that are wrong)")
    ax.set_title("Hit rate vs false-hit rate (calibration half)")
    ax.legend(frameon=False, fontsize=8)

    for ax, model in zip(axes[1:], MODELS):
        same, sibling = all_pairs[model]
        low = min(same.min(), sibling.min())
        high = max(same.max(), sibling.max())
        bins = np.linspace(low, high, 40)
        ax.hist(same, bins=bins, density=True, color="#444444", alpha=0.85, label="same topic")
        ax.hist(sibling, bins=bins, density=True, color="#cccccc", edgecolor="#888888",
                alpha=0.8, label="sibling topics (near-misses)")
        ax.set_xlabel("Similarity of a pair of queries")
        ax.set_ylabel("Density")
        ax.set_title(f"{model}\nAUC = {auc(same, sibling):.3f}")
        ax.legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(CHART_FILE, dpi=150)
    print(f"Chart saved in {CHART_FILE}")
    print(f"Spent in this run: ${log.total_cost() - spent_before:.4f}")


main()