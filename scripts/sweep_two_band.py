# Step B: the two-band cache (verifier in the middle) against the plain threshold.
# CALIBRATION half only. No API calls: it uses the saved embeddings and verdicts.
import csv
import json
import os
import sys

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from calibcache.config import CHEAP_MODEL, EMBEDDING_MODEL, STRONG_MODEL, VERIFIER_MODEL
from calibcache.dataset import load_queries
from calibcache.embed_queries import embed_queries
from calibcache.pricing import chat_cost, embedding_cost
from calibcache.simulate import sweep_two_band
from calibcache.usage_log import UsageLog

sys.stdout.reconfigure(encoding="utf-8")

LOW_MIN = 0.60   # the verifier checked every pair with similarity >= this
NUM_ORDERS = 20
LOS = [0.60, 0.65, 0.70, 0.75, 0.80]
HIS = [0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 1.01]    # 1.01 = never answer without checking
PLAIN_THRESHOLDS = [round(0.60 + 0.01 * k, 2) for k in range(40)]
TARGETS = [0.10, 0.05, 0.03, 0.02, 0.01]
MIN_HITS = 10
ANSWER_INPUT_TOKENS = 2050   # estimate: the policy prompt plus the question
ANSWER_OUTPUT_TOKENS = 100   # estimate
QUERY_TOKENS = 25            # estimate, for the embedding cost
VERDICTS_FILE = f"results/verdicts_{VERIFIER_MODEL}.json"
RESULTS_FILE = "results/two_band_calibration.csv"


def load_verdict_matrix(rows):
    """verdict[i, j] = 1 (YES), 0 (NO), -1 (unknown). i = new query, j = cached query."""
    index = {row["query_id"]: k for k, row in enumerate(rows)}
    verdict = np.full((len(rows), len(rows)), -1, dtype=np.int8)
    with open(VERDICTS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    costs = []
    seconds = []
    for key, item in data.items():
        new_id, cached_id = key.split("|")
        if new_id in index and cached_id in index:
            verdict[index[new_id], index[cached_id]] = 1 if item["yes"] else 0
            costs.append(item["cost"])
            seconds.append(item["seconds"])
    return verdict, float(np.mean(costs)), float(np.mean(seconds))


def add_savings(results, n, gen_cost, verifier_cost, embed_cost, name):
    """Money saved compared with answering every query with the model."""
    for r in results:
        misses = n - r["hits_mean"]
        with_cache = misses * gen_cost + r["calls_mean"] * verifier_cost + n * embed_cost
        r[name] = 1 - with_cache / (n * gen_cost)


def best_for_target(results, target):
    """Highest hit rate among settings whose false-hit rate is at most the target."""
    best = None
    for r in results:
        if r["hits_mean"] >= MIN_HITS and r["false_hit_pooled"] <= target:
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

    rows = [r for r in load_queries("data/queries_clean.jsonl") if r["split"] == "calibration"]
    n = len(rows)
    print(f"Calibration split only: {n} queries (the test split is not touched)")

    vectors = embed_queries(client, log, rows,
                            f"results/embeddings_calibration_{EMBEDDING_MODEL}.npz",
                            model=EMBEDDING_MODEL)
    sim = vectors @ vectors.T
    np.fill_diagonal(sim, -1)

    topic_names = sorted({r["intent_id"] for r in rows})
    labels = np.array([topic_names.index(r["intent_id"]) for r in rows])

    verdict, verifier_cost, verifier_seconds = load_verdict_matrix(rows)
    if ((sim >= LOW_MIN) & (verdict < 0)).any():
        raise SystemExit("Some verdicts are missing. Run scripts\\verify_pairs.py again.")

    orders = [np.random.default_rng(seed).permutation(n) for seed in range(NUM_ORDERS)]

    plain_configs = [(t, t) for t in PLAIN_THRESHOLDS]
    band_configs = [(lo, hi) for lo in LOS for hi in HIS if hi > lo]
    plain = sweep_two_band(sim, labels, orders, plain_configs, verdict)
    band = sweep_two_band(sim, labels, orders, band_configs, verdict)

    gen_cheap = chat_cost(CHEAP_MODEL, ANSWER_INPUT_TOKENS, ANSWER_OUTPUT_TOKENS)
    gen_strong = chat_cost(STRONG_MODEL, ANSWER_INPUT_TOKENS, ANSWER_OUTPUT_TOKENS)
    embed_cost = embedding_cost(EMBEDDING_MODEL, QUERY_TOKENS)
    for results in (plain, band):
        add_savings(results, n, gen_cheap, verifier_cost, embed_cost, "saving_vs_cheap")
        add_savings(results, n, gen_strong, verifier_cost, embed_cost, "saving_vs_strong")

    print(f"\nCosts used: answer with {CHEAP_MODEL} ${gen_cheap:.5f}, "
          f"with {STRONG_MODEL} ${gen_strong:.5f} (estimates), "
          f"verifier check ${verifier_cost:.6f} ({verifier_seconds:.2f} s, measured), "
          f"embedding ${embed_cost:.7f}")
    print(f"Results are averages over {NUM_ORDERS} random orders of the {n} queries.")

    # ---------- table 1: the two-band grid ----------
    print("\nTABLE 1: two-band settings (lo = below this: miss, hi = above this: hit, "
          "between: verifier)")
    print("false-hit% = wrong hits / hits (pooled over orders). worst = worst single order.")
    print(f"{'lo':>5} {'hi':>5} {'hit%':>6} {'false-hit%':>11} {'worst%':>7} "
          f"{'checks/100q':>12} {'save vs cheap':>14} {'save vs strong':>15}")
    for r in band:
        print(f"{r['lo']:>5.2f} {r['hi']:>5.2f} {100 * r['hit_rate']:>6.1f} "
              f"{100 * r['false_hit_pooled']:>11.1f} {100 * r['false_hit_worst']:>7.1f} "
              f"{100 * r['calls_mean'] / n:>12.1f} "
              f"{100 * r['saving_vs_cheap']:>13.1f}% {100 * r['saving_vs_strong']:>14.1f}%")

    # ---------- table 2: plain threshold vs two-band at the same false-hit level ----------
    print(f"\nTABLE 2: best hit rate at each false-hit level (needs at least {MIN_HITS} hits)")
    print("(chosen on this same data, so slightly optimistic; the test half gives the final numbers)")
    for target in TARGETS:
        print(f"\nfalse-hit <= {100 * target:.0f}%")
        p = best_for_target(plain, target)
        b = best_for_target(band, target)
        if p is None:
            print("  plain threshold : not reached")
        else:
            print(f"  plain threshold : hit {100 * p['hit_rate']:.1f}% at threshold {p['lo']:.2f}, "
                  f"false-hit {100 * p['false_hit_pooled']:.1f}%, "
                  f"saving vs strong {100 * p['saving_vs_strong']:.1f}%")
        if b is None:
            print("  two-band        : not reached")
        else:
            print(f"  two-band        : hit {100 * b['hit_rate']:.1f}% at lo {b['lo']:.2f}, "
                  f"hi {b['hi']:.2f}, false-hit {100 * b['false_hit_pooled']:.1f}%, "
                  f"worst order {100 * b['false_hit_worst']:.1f}%, "
                  f"checks {100 * b['calls_mean'] / n:.0f}/100 queries, "
                  f"saving vs strong {100 * b['saving_vs_strong']:.1f}%, "
                  f"vs cheap {100 * b['saving_vs_cheap']:.1f}%")

    # ---------- save everything for the charts ----------
    fields = ["kind", "lo", "hi", "hit_rate", "hits_mean", "wrong_mean", "false_hit_pooled",
              "false_hit_worst", "calls_mean", "main_hits", "main_wrong", "main_ci_high",
              "saving_vs_cheap", "saving_vs_strong"]
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for kind, results in (("plain", plain), ("two_band", band)):
            for r in results:
                writer.writerow({"kind": kind, **{k: r[k] for k in fields[1:]}})
    print(f"\nAll settings saved in {RESULTS_FILE}")


main()