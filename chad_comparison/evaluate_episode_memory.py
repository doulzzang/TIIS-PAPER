#!/usr/bin/env python3
"""Evaluate a deterministic episode-memory baseline on prepared CHAD windows.

Input is chad_windows_2s_4s.npz produced by prepare_chad_windows.py.
No test label is used for memory updates, score calibration, weight selection,
or threshold selection. Validation labels are used only to select component
weights and the reporting threshold.

Important: the current NPZ stores nearest-person relation features but not the
other person's ID. Therefore the "pair" component is actor-conditioned
pair-relation memory, not persistent pair-ID memory.
"""
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


def parse_id(value):
    video, actor, start, end = str(value).rsplit("|", 3)
    return video, actor, int(start), int(end)


def episode_embedding(x, feature_names, include_pair=False):
    mask = np.ones(len(feature_names), dtype=bool)
    if not include_pair:
        mask = np.array([not n.startswith("pair_") for n in feature_names])
    values = x[:, :, mask]
    mean = values.mean(axis=1)
    std = values.std(axis=1)
    delta = values[:, -1] - values[:, 0]
    return np.concatenate([mean, std, delta], axis=1).astype(np.float32)


def robust_fit(x):
    median = np.median(x, axis=0)
    q1, q3 = np.quantile(x, [.25, .75], axis=0)
    scale = q3 - q1
    scale[scale < 1e-5] = 1.0
    return median.astype(np.float32), scale.astype(np.float32)


def robust_transform(x, center, scale):
    z = (x - center) / scale
    return np.clip(np.nan_to_num(z, nan=0.0, posinf=20.0, neginf=-20.0),
                   -20.0, 20.0).astype(np.float32)


class DistanceCalibration:
    def __init__(self, reference):
        reference = np.asarray(reference, dtype=np.float64)
        self.median = float(np.median(reference))
        self.iqr = float(np.quantile(reference, .75) -
                         np.quantile(reference, .25))
        if self.iqr < EPS:
            self.iqr = max(float(np.std(reference)), 1.0)

    def __call__(self, value):
        z = np.clip((np.asarray(value) - self.median) / self.iqr, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))


def knn_reference(memory, k):
    model = NearestNeighbors(n_neighbors=min(k + 1, len(memory)),
                             metric="euclidean").fit(memory)
    distance, _ = model.kneighbors(memory)
    column = min(k, distance.shape[1] - 1)
    return model, distance[:, column]


def query_knn(model, query, k):
    distance, _ = model.kneighbors(query.reshape(1, -1),
                                   n_neighbors=min(k, model.n_samples_fit_))
    return float(distance[0].mean())


def history_distance(query, history):
    if not history:
        return np.nan
    matrix = np.asarray(history)
    distances = np.linalg.norm(matrix - query, axis=1)
    return float(np.mean(np.partition(
        distances, min(2, len(distances) - 1))[:min(3, len(distances))]))


def best_threshold(y, score):
    candidates = np.unique(np.quantile(score, np.linspace(.01, .99, 199)))
    values = [f1_score(y, score >= t, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(values))])


def metrics(y, score, threshold):
    pred = score >= threshold
    return {
        "episode_roc_auc": float(roc_auc_score(y, score)),
        "episode_pr_auc": float(average_precision_score(y, score)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "threshold": float(threshold),
        "n": int(len(y)),
        "positive": int(y.sum()),
    }


def component_scores(indices, metadata, actor_embed, pair_embed, global_model,
                     pair_model, global_cal, pair_cal, args):
    ordered = sorted(indices, key=lambda i: (
        metadata[i][0], metadata[i][2], metadata[i][1]))
    actor_history = defaultdict(lambda: deque(maxlen=args.memory_size))
    pair_history = defaultdict(lambda: deque(maxlen=args.memory_size))
    temporal = defaultdict(float)
    result = {}
    for i in ordered:
        video, actor, _, _ = metadata[i]
        key = (video, actor)
        global_raw = query_knn(global_model, actor_embed[i], args.k)
        global_score = float(global_cal(global_raw))

        actor_raw = history_distance(actor_embed[i], actor_history[key])
        actor_score = global_score if np.isnan(actor_raw) else float(global_cal(actor_raw))

        pair_global_raw = query_knn(pair_model, pair_embed[i], args.k)
        pair_global_score = float(pair_cal(pair_global_raw))
        pair_raw = history_distance(pair_embed[i], pair_history[key])
        pair_score = pair_global_score if np.isnan(pair_raw) else float(pair_cal(pair_raw))

        instant = (global_score + actor_score + pair_score) / 3.0
        previous = temporal[key]
        temporal_score = args.ema * previous + (1.0 - args.ema) * instant
        temporal[key] = temporal_score
        result[i] = [global_score, actor_score, pair_score, temporal_score]

        # Update online memory without consulting validation/test labels.
        if global_score <= args.update_gate:
            actor_history[key].append(actor_embed[i])
            pair_history[key].append(pair_embed[i])
    return np.asarray([result[i] for i in indices], dtype=np.float32)


def weight_grid(active, step):
    units = round(1.0 / step)
    for values in itertools.product(range(units + 1), repeat=len(active)):
        if sum(values) != units:
            continue
        # An ablation that names a component must actually use that component.
        if any(v == 0 for v in values):
            continue
        weights = np.zeros(4, dtype=np.float32)
        weights[list(active)] = np.asarray(values) / units
        yield weights


def select_weights(y, components, active, step):
    best = None
    for weights in weight_grid(active, step):
        score = components @ weights
        value = average_precision_score(y, score)
        if best is None or value > best[0] + EPS:
            best = (value, weights.copy())
    return best[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows", required=True)
    p.add_argument("--output-dir", default="results_episode_memory")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--memory-size", type=int, default=20)
    p.add_argument("--ema", type=float, default=.70)
    p.add_argument("--update-gate", type=float, default=.70)
    p.add_argument("--weight-step", type=float, default=.10)
    args = p.parse_args()

    data = np.load(args.windows, allow_pickle=False)
    x = np.nan_to_num(data["x"].astype(np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    y = data["y"].astype(np.int64)
    split = data["split"].astype(str)
    sample_id = data["sample_id"].astype(str)
    names = data["feature_names"].astype(str)
    metadata = [parse_id(v) for v in sample_id]

    actor = episode_embedding(x, names, include_pair=False)
    pair_columns = np.array([n.startswith("pair_") for n in names])
    if not pair_columns.any():
        raise ValueError("No pair_ features found in NPZ")
    pair_x = x[:, :, pair_columns]
    pair = np.concatenate([pair_x.mean(1), pair_x.std(1),
                           pair_x[:, -1] - pair_x[:, 0]], axis=1)

    train_normal = (split == "train") & (y == 0)
    ac, ass = robust_fit(actor[train_normal])
    pc, pss = robust_fit(pair[train_normal])
    actor = robust_transform(actor, ac, ass)
    pair = robust_transform(pair, pc, pss)

    global_model, global_reference = knn_reference(actor[train_normal], args.k)
    pair_model, pair_reference = knn_reference(pair[train_normal], args.k)
    global_cal = DistanceCalibration(global_reference)
    pair_cal = DistanceCalibration(pair_reference)

    val_idx = np.flatnonzero(split == "val")
    test_idx = np.flatnonzero(split == "test")
    val_components = component_scores(
        val_idx, metadata, actor, pair, global_model, pair_model,
        global_cal, pair_cal, args)
    test_components = component_scores(
        test_idx, metadata, actor, pair, global_model, pair_model,
        global_cal, pair_cal, args)

    variants = {
        "proposed_instant": (0,),
        "proposed_actor_memory": (0, 1),
        "proposed_actor_pair_relation_memory": (0, 1, 2),
        "proposed_full_episode_memory": (0, 1, 2, 3),
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    score_table = pd.DataFrame({
        "sample_id": sample_id[test_idx],
        "label": y[test_idx],
        "global_score": test_components[:, 0],
        "actor_score": test_components[:, 1],
        "pair_relation_score": test_components[:, 2],
        "temporal_score": test_components[:, 3],
    })
    selected = {}
    for name, active in variants.items():
        weights = select_weights(y[val_idx], val_components,
                                 active, args.weight_step)
        val_score = val_components @ weights
        test_score = test_components @ weights
        threshold = best_threshold(y[val_idx], val_score)
        row = {"model": name, **metrics(y[test_idx], test_score, threshold)}
        for component, weight in zip(
                ("w_global", "w_actor", "w_pair_relation", "w_temporal"), weights):
            row[component] = float(weight)
        rows.append(row)
        score_table[name + "_score"] = test_score
        selected[name] = weights.tolist()
        print(row)

    pd.DataFrame(rows).to_csv(out / "proposed_ablation_results.csv", index=False)
    score_table.to_csv(out / "proposed_test_scores.csv", index=False)
    with (out / "experiment_config.json").open("w", encoding="utf-8") as f:
        json.dump({**vars(args), "selected_weights": selected,
                   "pair_memory_limitation":
                   "NPZ has relation features but no persistent other-person ID."},
                  f, ensure_ascii=False, indent=2)
    print("Saved:", out.resolve())
    print("NOTE: pair component is pair-relation memory, not true pair-ID memory.")


if __name__ == "__main__":
    main()
