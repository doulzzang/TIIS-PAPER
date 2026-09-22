#!/usr/bin/env python3
"""CHAD actor/pair-ID episode-memory evaluation with leakage controls."""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.neighbors import NearestNeighbors

EPS = 1e-8


def parsed_metadata(data):
    sample_id = data["sample_id"].astype(str)
    if all(k in data.files for k in
           ("video_id", "actor_id", "pair_id", "start_frame", "end_frame")):
        return (sample_id, data["video_id"].astype(str),
                data["actor_id"].astype(str), data["pair_id"].astype(str),
                data["start_frame"].astype(int), data["end_frame"].astype(int))
    parsed = [str(v).rsplit("|", 3) for v in sample_id]
    print("WARNING: legacy NPZ has no pair_id; regenerate it with "
          "the revised prepare_chad_windows.py")
    return (sample_id, np.asarray([v[0] for v in parsed]),
            np.asarray([v[1] for v in parsed]),
            np.full(len(parsed), ""),
            np.asarray([int(v[2]) for v in parsed]),
            np.asarray([int(v[3]) for v in parsed]))


def embedding(x, names, pair_only=False):
    if pair_only:
        mask = np.asarray([n.startswith("pair_") for n in names])
    else:
        mask = np.asarray([not n.startswith("pair_") for n in names])
    values = x[:, :, mask]
    return np.concatenate([values.mean(1), values.std(1),
                           values[:, -1] - values[:, 0]], axis=1).astype(np.float32)


def robust_fit(x):
    center = np.median(x, axis=0)
    scale = np.quantile(x, .75, axis=0) - np.quantile(x, .25, axis=0)
    scale[scale < 1e-5] = 1.0
    return center.astype(np.float32), scale.astype(np.float32)


def robust_transform(x, center, scale):
    z = (x - center) / scale
    return np.clip(np.nan_to_num(z, nan=0, posinf=20, neginf=-20),
                   -20, 20).astype(np.float32)


class Calibrator:
    def __init__(self, values):
        values = np.asarray(values, dtype=float)
        self.center = float(np.median(values))
        self.scale = float(np.quantile(values, .75) -
                           np.quantile(values, .25))
        if self.scale < EPS:
            self.scale = max(float(np.std(values)), 1.0)

    def __call__(self, value):
        z = np.clip((np.asarray(value) - self.center) / self.scale, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))


def fit_knn(memory, k):
    model = NearestNeighbors(n_neighbors=min(k + 1, len(memory))).fit(memory)
    distance, _ = model.kneighbors(memory)
    ref = distance[:, min(k, distance.shape[1] - 1)]
    return model, ref


def query_knn(model, query, k):
    distance, _ = model.kneighbors(
        query.reshape(1, -1), n_neighbors=min(k, model.n_samples_fit_))
    return float(distance.mean())


def history_distance(query, history, nearest=3):
    matrix = np.asarray(history)
    distance = np.linalg.norm(matrix - query, axis=1)
    count = min(nearest, len(distance))
    return float(np.partition(distance, count - 1)[:count].mean())


def pair_key(video, actor, partner):
    if partner in ("", "None", "nan", "-1"):
        return None
    a, b = sorted((str(actor), str(partner)))
    return video, a, b


def history_calibration(indices, video, actor, pair, start, actor_z, pair_z,
                        min_history, memory_size):
    ah = defaultdict(lambda: deque(maxlen=memory_size))
    ph = defaultdict(lambda: deque(maxlen=memory_size))
    actor_values, pair_values = [], []
    ordered = sorted(indices, key=lambda i: (video[i], start[i], actor[i]))
    for i in ordered:
        akey = (video[i], actor[i])
        pkey = pair_key(video[i], actor[i], pair[i])
        if len(ah[akey]) >= min_history:
            actor_values.append(history_distance(actor_z[i], ah[akey]))
        if pkey is not None and len(ph[pkey]) >= min_history:
            pair_values.append(history_distance(pair_z[i], ph[pkey]))
        ah[akey].append(actor_z[i])
        if pkey is not None:
            ph[pkey].append(pair_z[i])
    return actor_values, pair_values


def score_split(indices, video, actor, pair, start, actor_z, pair_z,
                global_model, pair_model, calibrators, args):
    global_cal, actor_cal, pair_global_cal, pair_history_cal = calibrators
    ah = defaultdict(lambda: deque(maxlen=args.memory_size))
    ph = defaultdict(lambda: deque(maxlen=args.memory_size))
    ema = defaultdict(float)
    output = {}
    ordered = sorted(indices, key=lambda i: (video[i], start[i], actor[i]))
    for i in ordered:
        akey = (video[i], actor[i])
        pkey = pair_key(video[i], actor[i], pair[i])

        global_score = float(global_cal(
            query_knn(global_model, actor_z[i], args.k)))
        if len(ah[akey]) >= args.min_history:
            actor_score = float(actor_cal(
                history_distance(actor_z[i], ah[akey])))
        else:
            actor_score = global_score

        pair_global_score = float(pair_global_cal(
            query_knn(pair_model, pair_z[i], args.k)))
        if pkey is not None and len(ph[pkey]) >= args.min_history:
            pair_score = float(pair_history_cal(
                history_distance(pair_z[i], ph[pkey])))
        else:
            pair_score = pair_global_score

        instant = (global_score + actor_score + pair_score) / 3
        temporal_score = (args.ema * ema[akey] +
                          (1 - args.ema) * instant)
        ema[akey] = temporal_score
        output[i] = [global_score, actor_score, pair_score, temporal_score]

        # Gated online updates never inspect validation/test labels.
        if global_score <= args.update_gate:
            ah[akey].append(actor_z[i])
        if (pkey is not None and global_score <= args.update_gate and
                pair_global_score <= args.update_gate):
            ph[pkey].append(pair_z[i])
    return np.asarray([output[i] for i in indices], dtype=np.float32)


def weight_grid(active, step, maximum):
    if len(active) == 1:
        weight = np.zeros(4, dtype=np.float32)
        weight[active[0]] = 1
        yield weight
        return
    units = round(1 / step)
    max_units = round(maximum / step)
    for values in itertools.product(range(1, max_units + 1),
                                    repeat=len(active)):
        if sum(values) != units:
            continue
        weight = np.zeros(4, dtype=np.float32)
        weight[list(active)] = np.asarray(values) / units
        yield weight


def best_threshold(y, score):
    values = np.unique(np.quantile(score, np.linspace(.01, .99, 199)))
    f1 = [f1_score(y, score >= t, zero_division=0) for t in values]
    return float(values[int(np.argmax(f1))])


def choose_weights(y, components, active, args):
    best = None
    for weight in weight_grid(active, args.weight_step, args.max_weight):
        score = components @ weight
        pr = average_precision_score(y, score)
        threshold = best_threshold(y, score)
        f1 = f1_score(y, score >= threshold, zero_division=0)
        objective = args.pr_weight * pr + (1 - args.pr_weight) * f1
        if best is None or objective > best[0] + EPS:
            best = objective, weight.copy(), threshold
    if best is None:
        raise ValueError("No valid weight combination; check step/max-weight")
    return best[1], best[2]


def metrics(y, score, threshold):
    pred = score >= threshold
    return {
        "episode_roc_auc": float(roc_auc_score(y, score)),
        "episode_pr_auc": float(average_precision_score(y, score)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "threshold": float(threshold), "n": int(len(y)),
        "positive": int(y.sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows", required=True)
    p.add_argument("--output-dir", default="results_episode_memory_v2")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--memory-size", type=int, default=20)
    p.add_argument("--min-history", type=int, default=3)
    p.add_argument("--ema", type=float, default=.70)
    p.add_argument("--update-gate", type=float, default=.70)
    p.add_argument("--weight-step", type=float, default=.10)
    p.add_argument("--max-weight", type=float, default=.60)
    p.add_argument("--pr-weight", type=float, default=.70,
                   help="validation objective: value*PR-AUC + (1-value)*F1")
    args = p.parse_args()

    data = np.load(args.windows, allow_pickle=False)
    x = np.nan_to_num(data["x"].astype(np.float32),
                      nan=0, posinf=0, neginf=0)
    y = data["y"].astype(int)
    split = data["split"].astype(str)
    names = data["feature_names"].astype(str)
    sample_id, video, actor, pair, start, end = parsed_metadata(data)
    if np.all(pair == ""):
        raise ValueError("No pair IDs. Regenerate NPZ with revised "
                         "prepare_chad_windows.py before running v2.")

    actor_raw = embedding(x, names, False)
    pair_raw = embedding(x, names, True)
    train_normal = (split == "train") & (y == 0)
    ac, ass = robust_fit(actor_raw[train_normal])
    pc, pss = robust_fit(pair_raw[train_normal])
    actor_z = robust_transform(actor_raw, ac, ass)
    pair_z = robust_transform(pair_raw, pc, pss)

    global_model, global_ref = fit_knn(actor_z[train_normal], args.k)
    pair_model, pair_global_ref = fit_knn(pair_z[train_normal], args.k)
    normal_idx = np.flatnonzero(train_normal)
    actor_ref, pair_ref = history_calibration(
        normal_idx, video, actor, pair, start, actor_z, pair_z,
        args.min_history, args.memory_size)
    if not actor_ref or not pair_ref:
        raise RuntimeError("Insufficient normal histories for calibration")
    calibrators = (Calibrator(global_ref), Calibrator(actor_ref),
                   Calibrator(pair_global_ref), Calibrator(pair_ref))

    val_idx = np.flatnonzero(split == "val")
    test_idx = np.flatnonzero(split == "test")
    val_components = score_split(
        val_idx, video, actor, pair, start, actor_z, pair_z,
        global_model, pair_model, calibrators, args)
    test_components = score_split(
        test_idx, video, actor, pair, start, actor_z, pair_z,
        global_model, pair_model, calibrators, args)

    variants = {
        "proposed_instant": (0,),
        "proposed_actor_memory": (0, 1),
        "proposed_actor_pair_id_memory": (0, 1, 2),
        "proposed_full_episode_memory": (0, 1, 2, 3),
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    score_table = pd.DataFrame({
        "sample_id": sample_id[test_idx], "video_id": video[test_idx],
        "actor_id": actor[test_idx], "pair_id": pair[test_idx],
        "label": y[test_idx], "global_score": test_components[:, 0],
        "actor_score": test_components[:, 1],
        "pair_id_score": test_components[:, 2],
        "temporal_score": test_components[:, 3]})
    rows, selected = [], {}
    for name, active in variants.items():
        weight, threshold = choose_weights(
            y[val_idx], val_components, active, args)
        test_score = test_components @ weight
        row = {"model": name, **metrics(y[test_idx], test_score, threshold)}
        for key, value in zip(
                ("w_global", "w_actor", "w_pair_id", "w_temporal"), weight):
            row[key] = float(value)
        rows.append(row)
        score_table[name + "_score"] = test_score
        selected[name] = weight.tolist()
        print(row)

    pd.DataFrame(rows).to_csv(out / "proposed_ablation_results_v2.csv",
                              index=False)
    score_table.to_csv(out / "proposed_test_scores_v2.csv", index=False)
    with (out / "experiment_config_v2.json").open("w", encoding="utf-8") as f:
        json.dump({**vars(args), "selected_weights": selected,
                   "actor_history_calibration_n": len(actor_ref),
                   "pair_history_calibration_n": len(pair_ref)},
                  f, ensure_ascii=False, indent=2)
    print("Saved:", out.resolve())


if __name__ == "__main__":
    main()
