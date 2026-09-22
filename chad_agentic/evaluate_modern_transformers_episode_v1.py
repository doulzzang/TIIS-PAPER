#!/usr/bin/env python3
"""Apply the same actor-level episode evaluation to saved modern-model test scores.

Expected:
  results-dir/per_seed_results.csv
  results-dir/test_scores_<model>_seed<seed>.csv

Each score CSV must contain:
  sample_id,label,score
where sample_id is video_id|actor_id|start_frame|end_frame.

Thresholds are read from per_seed_results.csv (validation-selected upstream).
No test-set threshold tuning is performed here.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

EPS = 1e-12


def parse_sample_id(s):
    parts = str(s).rsplit("|", 3)
    if len(parts) != 4:
        raise ValueError(f"Unexpected sample_id: {s}")
    video, actor, start, end = parts
    return str(video), str(actor), int(start), int(end)


def build_episodes(df, positive_col, gap_frames, score_col=None):
    rows = []
    pos = df[df[positive_col].astype(int) == 1].copy()
    if pos.empty:
        return pd.DataFrame(columns=[
            "video_id", "actor_id", "start_frame", "end_frame",
            "window_count", "max_score", "mean_score"])
    for (video, actor), g in pos.groupby(
            ["video_id", "actor_id"], sort=False, dropna=False):
        g = g.sort_values(["start_frame", "end_frame"])
        cur = None
        for row in g.itertuples(index=False):
            start, end = int(row.start_frame), int(row.end_frame)
            score = float(getattr(row, score_col)) if score_col else np.nan
            if cur is None or start > cur["end_frame"] + gap_frames:
                if cur is not None:
                    rows.append(cur)
                cur = dict(video_id=str(video), actor_id=str(actor),
                           start_frame=start, end_frame=end,
                           window_count=1, scores=[score])
            else:
                cur["end_frame"] = max(cur["end_frame"], end)
                cur["window_count"] += 1
                cur["scores"].append(score)
        if cur is not None:
            rows.append(cur)
    for row in rows:
        scores = np.asarray(row.pop("scores"), dtype=float)
        row["max_score"] = float(np.nanmax(scores)) if np.isfinite(scores).any() else np.nan
        row["mean_score"] = float(np.nanmean(scores)) if np.isfinite(scores).any() else np.nan
    return pd.DataFrame(rows)


def match_episodes(gt, pred, tolerance, fps):
    matches, used_pred = [], set()
    for gi, g in gt.sort_values("start_frame").iterrows():
        candidates = pred[
            (pred["video_id"] == g.video_id) &
            (pred["actor_id"] == g.actor_id) &
            (pred["end_frame"] >= g.start_frame - tolerance) &
            (pred["start_frame"] <= g.end_frame + tolerance)]
        candidates = candidates.loc[~candidates.index.isin(used_pred)].copy()
        if candidates.empty:
            continue
        candidates["distance"] = np.maximum(
            0, np.maximum(g.start_frame - candidates["end_frame"],
                          candidates["start_frame"] - g.end_frame))
        candidates["onset_distance"] = abs(
            candidates["start_frame"] - g.start_frame)
        pi = candidates.sort_values(
            ["distance", "onset_distance", "start_frame"]).index[0]
        p = pred.loc[pi]
        used_pred.add(pi)
        matches.append({
            "gt_episode_index": int(gi),
            "pred_episode_index": int(pi),
            "video_id": g.video_id,
            "actor_id": g.actor_id,
            "gt_start_frame": int(g.start_frame),
            "gt_end_frame": int(g.end_frame),
            "pred_start_frame": int(p.start_frame),
            "pred_end_frame": int(p.end_frame),
            "delay_frames": int(p.start_frame - g.start_frame),
            "delay_sec": float((p.start_frame - g.start_frame) / fps),
        })
    return pd.DataFrame(matches), used_pred


def evaluate(score_file, threshold, gap_frames, tolerance, fps):
    df = pd.read_csv(score_file)
    needed = {"sample_id", "label", "score"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{score_file}: missing {sorted(needed-set(df.columns))}")

    parsed = [parse_sample_id(s) for s in df["sample_id"]]
    df["video_id"] = [v[0] for v in parsed]
    df["actor_id"] = [v[1] for v in parsed]
    df["start_frame"] = [v[2] for v in parsed]
    df["end_frame"] = [v[3] for v in parsed]
    df["prediction"] = (df["score"].astype(float) >= threshold).astype(np.int8)

    gt = build_episodes(df, "label", gap_frames)
    pred = build_episodes(df, "prediction", gap_frames, "score")
    matches, used = match_episodes(gt, pred, tolerance, fps)

    precision = len(used) / max(len(pred), 1)
    recall = len(matches) / max(len(gt), 1)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    delay = matches["delay_sec"] if not matches.empty else pd.Series(dtype=float)

    return {
        "gt_episodes": int(len(gt)),
        "pred_episodes": int(len(pred)),
        "matched_gt_episodes": int(len(matches)),
        "false_alarm_episodes": int(len(pred) - len(used)),
        "episode_precision": float(precision),
        "episode_recall": float(recall),
        "episode_f1": float(f1),
        "mean_delay_sec": float(delay.mean()) if len(delay) else np.nan,
        "median_delay_sec": float(delay.median()) if len(delay) else np.nan,
        "early_or_on_time_rate": float((delay <= 0).mean()) if len(delay) else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_modern_transformers")
    ap.add_argument("--output-dir", default="results_modern_transformers_episode")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--gap-sec", type=float, default=1.05)
    ap.add_argument("--match-tolerance-sec", type=float, default=0.0)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    seed_table = pd.read_csv(results_dir / "per_seed_results.csv")
    gap_frames = int(round(args.gap_sec * args.fps))
    tolerance = int(round(args.match_tolerance_sec * args.fps))

    rows = []
    for r in seed_table.itertuples(index=False):
        model, seed = str(r.model), int(r.seed)
        threshold = float(r.threshold)
        score_file = results_dir / f"test_scores_{model}_seed{seed}.csv"
        if not score_file.exists():
            print("SKIP missing:", score_file)
            continue
        em = evaluate(score_file, threshold, gap_frames, tolerance, args.fps)
        row = {
            "model": model, "seed": seed, "threshold": threshold,
            "window_roc_auc": float(r.episode_roc_auc),
            "window_pr_auc": float(r.episode_pr_auc),
            "window_f1": float(r.f1),
            "window_precision": float(r.precision),
            "window_recall": float(r.recall),
            **em
        }
        rows.append(row)
        print(row)

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out / "episode_per_seed_results.csv", index=False)

    metrics = [
        "window_roc_auc", "window_pr_auc", "window_f1",
        "window_precision", "window_recall",
        "episode_precision", "episode_recall", "episode_f1",
        "false_alarm_episodes", "mean_delay_sec",
        "median_delay_sec", "early_or_on_time_rate"
    ]
    summary_rows = []
    for model, g in per_seed.groupby("model", sort=False):
        row = {"model": model, "seeds": int(len(g))}
        for m in metrics:
            row[m + "_mean"] = float(g[m].mean())
            row[m + "_std"] = float(g[m].std(ddof=1)) if len(g) > 1 else 0.0
        # These should be invariant across seeds; retain for sanity checking.
        row["gt_episodes"] = int(g["gt_episodes"].iloc[0])
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "episode_summary_results.csv", index=False)

    print("\n=== 3-SEED EPISODE SUMMARY ===")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(summary.to_string(index=False))
    print("\nSaved:", out.resolve())


if __name__ == "__main__":
    main()
