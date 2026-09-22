#!/usr/bin/env python3
"""Evaluate actor and pair grounding on manually annotated CHAD segments."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_sample(value):
    video, actor, start, end = str(value).rsplit("|", 3)
    return video, actor, int(start), int(end)


def split_ids(value):
    if pd.isna(value) or str(value).strip() == "":
        return set()
    return {x.strip() for x in str(value).split(";") if x.strip()}


def canonical_pair(value):
    if pd.isna(value) or str(value).strip() == "":
        return ""
    parts = str(value).split("-")
    return "-".join(sorted(parts)) if len(parts) == 2 else str(value)


def reciprocal_rank(ranking, truths):
    for rank, item in enumerate(ranking, 1):
        if item in truths:
            return 1.0 / rank
    return 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ground-truth", default="actor_pair_ground_truth.csv")
    p.add_argument("--scores",
                   default="results_episode_memory_v2/proposed_test_scores_v2.csv")
    p.add_argument("--output-dir", default="results_actor_pair_grounding")
    p.add_argument("--aggregation", choices=["max", "mean"], default="max")
    args = p.parse_args()

    gt = pd.read_csv(args.ground_truth)
    scores = pd.read_csv(args.scores)
    required_gt = {"video_id", "start_frame", "end_frame",
                   "abnormal_actor_ids", "abnormal_pair_ids"}
    required_score = {"sample_id", "actor_id", "pair_id",
                      "global_score", "actor_score", "pair_id_score"}
    if not required_gt.issubset(gt.columns):
        raise ValueError("Ground truth missing: " +
                         str(sorted(required_gt - set(gt.columns))))
    if not required_score.issubset(scores.columns):
        raise ValueError("Score file missing: " +
                         str(sorted(required_score - set(scores.columns))))

    parsed = [parse_sample(v) for v in scores.sample_id]
    scores["video_from_sample"] = [v[0] for v in parsed]
    scores["window_start"] = [v[2] for v in parsed]
    scores["window_end"] = [v[3] for v in parsed]
    scores["actor_id"] = scores.actor_id.astype(str)
    scores["pair_id"] = scores.pair_id.fillna("").astype(str)
    scores["pair_key"] = [
        "-".join(sorted((a, b))) if b not in ("", "nan", "None", "-1") else ""
        for a, b in zip(scores.actor_id, scores.pair_id)]

    details = []
    for _, row in gt.iterrows():
        video = str(row.video_id)
        start, end = int(row.start_frame), int(row.end_frame)
        subset = scores[
            (scores.video_from_sample == video) &
            (scores.window_end >= start) &
            (scores.window_start <= end)
        ].copy()
        if subset.empty:
            details.append({
                "video_id": video, "start_frame": start, "end_frame": end,
                "status": "no_matching_score"})
            continue
        agg = "max" if args.aggregation == "max" else "mean"
        actor_table = subset.groupby("actor_id").agg(
            instant=("global_score", agg),
            grounded=("actor_score", agg)).reset_index()
        pair_table = subset[subset.pair_key != ""].groupby("pair_key").agg(
            pair_grounded=("pair_id_score", agg)).reset_index()

        true_actors = split_ids(row.abnormal_actor_ids)
        true_pair = canonical_pair(row.abnormal_pair_ids)
        instant_rank = actor_table.sort_values(
            "instant", ascending=False).actor_id.tolist()
        actor_rank = actor_table.sort_values(
            "grounded", ascending=False).actor_id.tolist()
        pair_rank = pair_table.sort_values(
            "pair_grounded", ascending=False).pair_key.tolist()
        details.append({
            "video_id": video, "start_frame": start, "end_frame": end,
            "event_type": row.get("event_type", ""),
            "status": "ok",
            "true_actor_ids": ";".join(sorted(true_actors)),
            "true_pair": true_pair,
            "instant_actor_top1": instant_rank[0] if instant_rank else "",
            "grounded_actor_top1": actor_rank[0] if actor_rank else "",
            "grounded_pair_top1": pair_rank[0] if pair_rank else "",
            "instant_actor_hit1": int(bool(instant_rank) and
                                      instant_rank[0] in true_actors),
            "grounded_actor_hit1": int(bool(actor_rank) and
                                       actor_rank[0] in true_actors),
            "grounded_pair_hit1": int(bool(true_pair) and bool(pair_rank) and
                                      pair_rank[0] == true_pair),
            "instant_actor_mrr": reciprocal_rank(instant_rank, true_actors),
            "grounded_actor_mrr": reciprocal_rank(actor_rank, true_actors),
            "grounded_pair_mrr": reciprocal_rank(
                pair_rank, {true_pair} if true_pair else set()),
            "actor_candidates": len(actor_rank),
            "pair_candidates": len(pair_rank),
        })

    detail = pd.DataFrame(details)
    valid = detail[detail.status == "ok"].copy()
    if valid.empty:
        raise RuntimeError("No annotated segment matched the score file")
    pair_valid = valid[valid.true_pair.astype(str) != ""]
    summary = pd.DataFrame([
        {"metric": "instant_actor_hit_at_1",
         "value": valid.instant_actor_hit1.mean(), "n": len(valid)},
        {"metric": "grounded_actor_hit_at_1",
         "value": valid.grounded_actor_hit1.mean(), "n": len(valid)},
        {"metric": "instant_actor_mrr",
         "value": valid.instant_actor_mrr.mean(), "n": len(valid)},
        {"metric": "grounded_actor_mrr",
         "value": valid.grounded_actor_mrr.mean(), "n": len(valid)},
        {"metric": "grounded_pair_hit_at_1",
         "value": pair_valid.grounded_pair_hit1.mean()
         if len(pair_valid) else np.nan, "n": len(pair_valid)},
        {"metric": "grounded_pair_mrr",
         "value": pair_valid.grounded_pair_mrr.mean()
         if len(pair_valid) else np.nan, "n": len(pair_valid)},
    ])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "actor_pair_grounding_details.csv", index=False)
    summary.to_csv(out / "actor_pair_grounding_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("Saved:", out.resolve())


if __name__ == "__main__":
    main()
