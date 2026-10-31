# Simulates a cache that starts empty and receives a stream of queries.
import math

import numpy as np

MISS, CORRECT, WRONG = 0, 1, 2


def wilson_interval(successes, total, z=1.96):
    """95% confidence interval for a proportion (Wilson method). Returns (low, high)."""
    if total == 0:
        return 0.0, 1.0
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def run_stream(sim, labels, order, threshold):
    """Replay the queries in the given order into an empty cache (plain threshold).
    sim    : similarity matrix (every query against every query)
    labels : topic number of every query
    Returns (status, matched):
      status[i]  = MISS (0), CORRECT hit (1) or WRONG hit (2)
      matched[i] = index of the cached query that was used (-1 if miss)"""
    n = len(labels)
    status = np.zeros(n, dtype=np.int8)
    matched = np.full(n, -1, dtype=np.int32)
    cache = []  # indexes of the queries stored in the cache

    for i in order:
        if cache:
            scores = sim[i, cache]
            best = int(np.argmax(scores))
            if scores[best] >= threshold:
                j = cache[best]
                matched[i] = j
                status[i] = CORRECT if labels[j] == labels[i] else WRONG
                continue
        cache.append(i)  # miss: ask the model, then store the query

    return status, matched


def ceiling_hit_rate(labels, order):
    """Best possible hit rate: queries whose topic already appeared earlier."""
    seen = set()
    possible = 0
    for i in order:
        if labels[i] in seen:
            possible += 1
        seen.add(labels[i])
    return possible / len(order)


def sweep(sim, labels, is_near_miss, thresholds, orders):
    """For every threshold, replay all the orders. The main order is orders[0].
    Returns a list with one dictionary of results per threshold."""
    n = len(labels)
    results = []
    for t in thresholds:
        rates = []  # false-hit rate in each of the orders
        for k, order in enumerate(orders):
            status, matched = run_stream(sim, labels, order, t)
            h = int((status > 0).sum())
            w = int((status == WRONG).sum())
            if h > 0:
                rates.append(w / h)
            if k == 0:  # the main order
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
    return results


def auc(positive, negative):
    """Chance that a random 'positive' score is higher than a random 'negative' score.
    1.0 = perfect separation, 0.5 = no better than a coin flip."""
    scores = np.concatenate([positive, negative])
    order = scores.argsort()
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    rank_sum = ranks[:len(positive)].sum()
    n_pos, n_neg = len(positive), len(negative)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def pair_groups(sim, labels, family_labels):
    """Similarity scores of all pairs of queries, in three groups:
      same    : same topic
      sibling : different topic, same family (the near-misses)
      other   : different family"""
    i, j = np.triu_indices(len(labels), k=1)
    scores = sim[i, j]
    same_topic = labels[i] == labels[j]
    same_family = family_labels[i] == family_labels[j]
    same = scores[same_topic]
    sibling = scores[(~same_topic) & same_family]
    other = scores[~same_family]
    return same, sibling, other


def run_stream_two_band(sim, labels, order, lo, hi, verdict):
    """Replay the queries into an empty cache with the two-band rule:
      best score >= hi        -> hit
      lo <= best score < hi   -> ask the verifier (saved verdict decides)
      best score < lo         -> miss
    verdict[i, j] = 1 (YES), 0 (NO) or -1 (unknown): new query i, cached query j.
    With lo == hi there is no middle band, so this is the plain threshold.
    Returns (status, number of verifier calls)."""
    n = len(labels)
    status = np.zeros(n, dtype=np.int8)
    calls = 0
    cache = []

    for i in order:
        if cache:
            scores = sim[i, cache]
            best = int(np.argmax(scores))
            score = scores[best]
            j = cache[best]
            hit = False
            if score >= hi:
                hit = True
            elif score >= lo:
                calls += 1
                v = verdict[i, j]
                if v < 0:
                    raise ValueError("A verifier verdict is missing. Run verify_pairs.py again.")
                hit = (v == 1)
            if hit:
                status[i] = CORRECT if labels[j] == labels[i] else WRONG
                continue
        cache.append(i)  # miss: ask the model, then store the query

    return status, calls


def sweep_two_band(sim, labels, orders, configs, verdict):
    """For every (lo, hi), replay all the orders. The main order is orders[0].
    Returns a list with one dictionary of results per (lo, hi)."""
    n = len(labels)
    results = []
    for lo, hi in configs:
        hits_list, wrong_list, calls_list, rates = [], [], [], []
        main_hits = main_wrong = 0
        for k, order in enumerate(orders):
            status, calls = run_stream_two_band(sim, labels, order, lo, hi, verdict)
            h = int((status > 0).sum())
            w = int((status == WRONG).sum())
            hits_list.append(h)
            wrong_list.append(w)
            calls_list.append(calls)
            if h > 0:
                rates.append(w / h)
            if k == 0:
                main_hits, main_wrong = h, w

        total_hits = sum(hits_list)
        total_wrong = sum(wrong_list)
        _, ci_high = wilson_interval(main_wrong, main_hits)
        results.append({
            "lo": lo,
            "hi": hi,
            "hit_rate": float(np.mean(hits_list)) / n,
            "hits_mean": float(np.mean(hits_list)),
            "wrong_mean": float(np.mean(wrong_list)),
            "false_hit_pooled": total_wrong / total_hits if total_hits else 0.0,
            "false_hit_worst": max(rates) if rates else 0.0,
            "calls_mean": float(np.mean(calls_list)),
            "main_hits": main_hits,
            "main_wrong": main_wrong,
            "main_ci_high": ci_high,
        })
    return results