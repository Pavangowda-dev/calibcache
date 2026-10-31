# Threshold sweep on the CALIBRATION half only.
# For each similarity threshold: hit rate and false-hit rate.
# The test half is not touched here.
import csv
import os
import sys

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from calibcache.dataset import load_queries
from calibcache.embed_queries import embed_queries
from calibcache.simulate import WRONG, ceiling_hit_rate, run_stream, wilson_interval
from calibcache.usage_log import UsageLog

sys.stdout.reconfigure(encoding="utf-8")

THRESHOLDS = [round(0.60 + 0.01 * k, 2) for k in range(40)]   # 0.60 ... 0.99
PRINT_THRESHOLDS = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84,
                    0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
NUM_ORDERS = 20                 # random orders of the stream
PAIR_THRESHOLDS = [0.80, 0.90]  # where we look at which topics get mixed up
TARGETS = [0.05, 0.02, 0.01]    # false-hit targets for the preview
RESULTS_FILE = "results/sweep_calibration.csv"


def pct(part, total):
    """Percentage as text, or '-' if there is nothing to divide."""
    return "-" if total == 0 else f"{100 * part / total:.1f}"


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.startswith("sk-"):
        raise SystemExit("OPENAI_API_KEY is missing or wrong in .env")

    client = OpenAI(api_key=api_key)
    log = UsageLog()

    # Only the calibration split
    rows = [r for r in load_queries("data/queries_clean.jsonl") if r["split"] == "calibration"]
    n = len(rows)
    print(f"Calibration split only: {n} queries (the test split is not touched)")

    vectors = embed_queries(client, log, rows, "results/embeddings_calibration.npz")
    sim = vectors @ vectors.T  # similarity of every query with every other query

    topic_names = sorted({r["intent_id"] for r in rows})
    topic_index = {name: k for k, name in enumerate(topic_names)}
    labels = np.array([topic_index[r["intent_id"]] for r in rows])
    is_near_miss = np.array([r["style"] == "minimal_edit" for r in rows])
    print(f"Topics: {len(topic_names)}, near-miss (minimal_edit) queries: {int(is_near_miss.sum())}")

    orders = [np.random.default_rng(seed).permutation(n) for seed in range(NUM_ORDERS)]
    print(f"Best possible hit rate in the main order: "
          f"{100 * ceiling_hit_rate(labels, orders[0]):.1f}%\n")

    # ---------- the sweep ----------
    results = []
    main_runs = {}
    for t in THRESHOLDS:
        rates = []  # false-hit rate in each of the random orders
        for k, order in enumerate(orders):
            status, matched = run_stream(sim, labels, order, t)
            h = int((status > 0).sum())
            w = int((status == WRONG).sum())
            if h > 0:
                rates.append(w / h)
            if k == 0:  # the main order
                main_runs[t] = (status, matched)
                hits, wrong = h, w
                nm_hits = int(((status > 0) & is_near_miss).sum())
                nm_wrong = int(((status == WRONG) & is_near_miss).sum())

        low, high = wilson_interval(wrong, hits)
        results.append({
            "threshold": t,
            "hit_rate": hits / n,
            "hits": hits,
            "wrong": wrong,
            "false_hit_rate": wrong / hits if hits else 0.0,
            "ci_low": low,
            "ci_high": high,
            "near_miss_hits": nm_hits,
            "near_miss_wrong": nm_wrong,
            "other_hits": hits - nm_hits,
            "other_wrong": wrong - nm_wrong,
            "orders_false_hit_min": min(rates) if rates else 0.0,
            "orders_false_hit_max": max(rates) if rates else 0.0,
        })

    # ---------- table 1: all queries ----------
    print("TABLE 1: all queries (main order)")
    print("false-hit% = wrong hits / all hits. upper95 = top of the 95% interval.")
    print(f"{'thr':>5} {'hit%':>6} {'hits':>5} {'wrong':>5} {'false-hit%':>11} "
          f"{'upper95%':>9}   range over {NUM_ORDERS} orders")
    for r in results:
        if r["threshold"] in PRINT_THRESHOLDS:
            print(f"{r['threshold']:>5.2f} {100 * r['hit_rate']:>6.1f} {r['hits']:>5} "
                  f"{r['wrong']:>5} {100 * r['false_hit_rate']:>11.1f} "
                  f"{100 * r['ci_high']:>9.1f}   "
                  f"{100 * r['orders_false_hit_min']:.1f} - {100 * r['orders_false_hit_max']:.1f}%")

    # ---------- table 2: near-miss queries vs the others ----------
    print(f"\nTABLE 2: near-miss queries ({int(is_near_miss.sum())}) vs all other queries "
          f"({n - int(is_near_miss.sum())}), main order")
    print(f"{'thr':>5} | {'NM hits':>7} {'NM wrong':>8} {'NM false-hit%':>13} | "
          f"{'other hits':>10} {'other wrong':>11} {'other false-hit%':>16}")
    for r in results:
        if r["threshold"] in PRINT_THRESHOLDS:
            print(f"{r['threshold']:>5.2f} | {r['near_miss_hits']:>7} {r['near_miss_wrong']:>8} "
                  f"{pct(r['near_miss_wrong'], r['near_miss_hits']):>13} | "
                  f"{r['other_hits']:>10} {r['other_wrong']:>11} "
                  f"{pct(r['other_wrong'], r['other_hits']):>16}")

    # ---------- which topics get mixed up ----------
    for t in PAIR_THRESHOLDS:
        status, matched = main_runs[t]
        pairs = {}
        for i in np.where(status == WRONG)[0]:
            key = (topic_names[labels[matched[i]]], topic_names[labels[i]])
            pairs[key] = pairs.get(key, 0) + 1
        print(f"\nMost common wrong hits at threshold {t:.2f} (cached topic -> asked topic):")
        ranked = sorted(pairs.items(), key=lambda item: item[1], reverse=True)
        if not ranked:
            print("  none")
        for (cached, asked), count in ranked[:6]:
            print(f"  {count:>3}  {cached} -> {asked}")

    # ---------- preview of the calibration idea ----------
    print("\nPREVIEW (not final: chosen on the same data it is measured on):")
    print("Lowest threshold whose 95% upper bound of the false-hit rate is under the target:")
    for target in TARGETS:
        found = None
        for r in results:
            if r["hits"] > 0 and r["ci_high"] <= target:
                found = r
                break
        if found is None:
            print(f"  target {100 * target:.0f}%: no threshold reaches it")
        else:
            print(f"  target {100 * target:.0f}%: threshold {found['threshold']:.2f} -> "
                  f"hit rate {100 * found['hit_rate']:.1f}% "
                  f"({found['hits']} hits, {found['wrong']} wrong)")

    # ---------- save every threshold for the charts later ----------
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nAll thresholds saved in {RESULTS_FILE}")


main()