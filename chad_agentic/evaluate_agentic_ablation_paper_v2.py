#!/usr/bin/env python3
"""Episode-level evaluation for CHAD Agentic window predictions.

This script merges consecutive window labels/alerts for each video and actor,
then performs one-to-one temporal matching. Detection delay is measured from
the first positive *prediction-window* of a ground-truth episode; it is not the
raw-video abnormal-action onset unless raw frame annotations are supplied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12


def clean_id(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).replace({"nan": "", "None": ""})


def build_episodes(df: pd.DataFrame, positive_col: str, gap_frames: int,
                   score_col: str | None = None) -> pd.DataFrame:
    rows = []
    positive = df[df[positive_col].astype(int) == 1].copy()
    if positive.empty:
        return pd.DataFrame(columns=[
            "video_id", "actor_id", "start_frame", "end_frame",
            "window_count", "max_score", "mean_score"])
    for (video, actor), group in positive.groupby(
            ["video_id", "actor_id"], sort=False, dropna=False):
        group = group.sort_values(["start_frame", "end_frame"])
        current = None
        for row in group.itertuples(index=False):
            start, end = int(row.start_frame), int(row.end_frame)
            score = float(getattr(row, score_col)) if score_col else np.nan
            if current is None or start > current["end_frame"] + gap_frames:
                if current is not None:
                    rows.append(current)
                current = {"video_id": str(video), "actor_id": str(actor),
                           "start_frame": start, "end_frame": end,
                           "window_count": 1, "scores": [score]}
            else:
                current["end_frame"] = max(current["end_frame"], end)
                current["window_count"] += 1
                current["scores"].append(score)
        if current is not None:
            rows.append(current)
    for row in rows:
        scores = np.asarray(row.pop("scores"), dtype=float)
        row["max_score"] = (float(np.nanmax(scores))
                            if np.isfinite(scores).any() else np.nan)
        row["mean_score"] = (float(np.nanmean(scores))
                             if np.isfinite(scores).any() else np.nan)
    return pd.DataFrame(rows)


def match_episodes(gt: pd.DataFrame, pred: pd.DataFrame, tolerance: int,
                   fps: float):
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
            "gt_episode_index": int(gi), "pred_episode_index": int(pi),
            "video_id": g.video_id, "actor_id": g.actor_id,
            "gt_start_frame": int(g.start_frame),
            "gt_end_frame": int(g.end_frame),
            "pred_start_frame": int(p.start_frame),
            "pred_end_frame": int(p.end_frame),
            "delay_frames": int(p.start_frame - g.start_frame),
            "delay_sec": float((p.start_frame - g.start_frame) / fps)})
    return pd.DataFrame(matches), used_pred


def evaluate_model(df, gt, model, score_col, threshold, gap_frames,
                   tolerance, fps, total_hours):
    pred_col = "__prediction"
    work = df.copy()
    if model == "proposed_agentic_state_memory" and "agent_alert" in work:
        work[pred_col] = work["agent_alert"].astype(np.int8)
    else:
        work[pred_col] = (
            work[score_col].astype(float) >= threshold).astype(np.int8)
    pred = build_episodes(work, pred_col, gap_frames, score_col)
    matches, used_pred = match_episodes(gt, pred, tolerance, fps)
    tp_gt, tp_pred = len(matches), len(used_pred)
    recall = tp_gt / max(len(gt), 1)
    precision = tp_pred / max(len(pred), 1)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    false_episodes = len(pred) - tp_pred
    delay = matches["delay_sec"] if not matches.empty else pd.Series(dtype=float)
    result = {
        "model": model,
        "threshold": ("agent_state_equals_ALERT"
                      if model == "proposed_agentic_state_memory"
                      and "agent_alert" in work else float(threshold)),
        "gt_episodes": int(len(gt)), "pred_episodes": int(len(pred)),
        "matched_gt_episodes": int(tp_gt),
        "false_alarm_episodes": int(false_episodes),
        "episode_precision": float(precision),
        "episode_recall": float(recall), "episode_f1": float(f1),
        "false_alarms_per_hour": float(false_episodes / max(total_hours, EPS)),
        "mean_delay_sec": float(delay.mean()) if len(delay) else np.nan,
        "median_delay_sec": float(delay.median()) if len(delay) else np.nan,
        "early_or_on_time_rate": float((delay <= 0).mean()) if len(delay) else np.nan}
    pred = pred.copy()
    pred["matched"] = pred.index.isin(used_pred).astype(np.int8)
    return result, pred, matches



def make_paper_table(ablation: pd.DataFrame, episode_results: pd.DataFrame) -> pd.DataFrame:
    """Merge window-level ablation metrics with episode-level metrics."""
    keep_window = [
        "model", "roc_auc", "pr_auc", "f1", "precision", "recall", "threshold"
    ]
    available = [c for c in keep_window if c in ablation.columns]
    win = ablation[available].copy()

    rename_win = {
        "roc_auc": "window_roc_auc",
        "pr_auc": "window_pr_auc",
        "f1": "window_f1",
        "precision": "window_precision",
        "recall": "window_recall",
        "threshold": "validation_threshold",
    }
    win = win.rename(columns=rename_win)

    ep_cols = [
        "model", "gt_episodes", "pred_episodes", "matched_gt_episodes",
        "false_alarm_episodes", "episode_precision", "episode_recall",
        "episode_f1", "false_alarms_per_hour", "mean_delay_sec",
        "median_delay_sec", "early_or_on_time_rate"
    ]
    ep = episode_results[[c for c in ep_cols if c in episode_results.columns]].copy()

    paper = win.merge(ep, on="model", how="inner")

    display_names = {
        "proposed_instant": "Instant",
        "proposed_actor_memory": "+ Actor Memory",
        "proposed_actor_pair_id_memory": "+ Actor-Pair ID Memory",
        "proposed_full_episode_memory": "+ Full Episode Memory",
        "proposed_agentic_state_memory": "+ Agentic State Memory",
    }
    paper.insert(0, "method", paper["model"].map(display_names).fillna(paper["model"]))

    order = list(display_names)
    paper["_order"] = paper["model"].map({m: i for i, m in enumerate(order)})
    paper = paper.sort_values("_order").drop(columns="_order")

    preferred = [
        "method", "model",
        "window_roc_auc", "window_pr_auc", "window_f1",
        "window_precision", "window_recall", "validation_threshold",
        "gt_episodes", "pred_episodes", "matched_gt_episodes",
        "episode_precision", "episode_recall", "episode_f1",
        "false_alarm_episodes", "false_alarms_per_hour",
        "mean_delay_sec", "median_delay_sec", "early_or_on_time_rate"
    ]
    return paper[[c for c in preferred if c in paper.columns]]


def save_latex_table(paper: pd.DataFrame, path: Path) -> None:
    """Save a compact IEEE-style LaTeX table fragment."""
    cols = [
        "method", "window_roc_auc", "window_pr_auc", "window_f1",
        "episode_precision", "episode_recall", "episode_f1",
        "false_alarm_episodes", "mean_delay_sec"
    ]
    table = paper[[c for c in cols if c in paper.columns]].copy()
    numeric_cols = table.select_dtypes(include=[np.number]).columns
    table[numeric_cols] = table[numeric_cols].round(3)
    latex = table.to_latex(index=False, escape=True, na_rep="--")
    path.write_text(latex, encoding="utf-8")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True,
                   help="agentic_test_scores_and_reasons.csv")
    p.add_argument("--ablation", required=True,
                   help="agentic_ablation_results.csv with validation thresholds")
    p.add_argument("--output-dir", default="results_episode_level")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--gap-sec", type=float, default=1.05,
                   help="maximum separation when merging adjacent windows")
    p.add_argument("--match-tolerance-sec", type=float, default=0.0)
    args = p.parse_args()

    df = pd.read_csv(args.scores)
    required = {"video_id", "actor_id", "start_frame", "end_frame", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing score-table columns: {sorted(missing)}")
    df["video_id"], df["actor_id"] = clean_id(df.video_id), clean_id(df.actor_id)
    df = df.sort_values(["video_id", "actor_id", "start_frame"]).reset_index(drop=True)
    gap_frames = int(round(args.gap_sec * args.fps))
    tolerance = int(round(args.match_tolerance_sec * args.fps))
    gt = build_episodes(df, "label", gap_frames)

    # Approximate observed duration from each video's frame range. This supports
    # comparable false-alarm/hour reporting on the prediction-window timeline.
    span = df.groupby("video_id").agg(
        first=("start_frame", "min"), last=("end_frame", "max"))
    total_hours = float(((span["last"] - span["first"] + 1).clip(lower=1)
                         / args.fps).sum() / 3600)

    ablation = pd.read_csv(args.ablation)
    score_columns = {
        "proposed_instant": "proposed_instant_score",
        "proposed_actor_memory": "proposed_actor_memory_score",
        "proposed_actor_pair_id_memory": "proposed_actor_pair_id_memory_score",
        "proposed_full_episode_memory": "proposed_full_episode_memory_score",
        "proposed_agentic_state_memory":
            "proposed_agentic_state_memory_score"}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for row in ablation.itertuples(index=False):
        model = str(row.model)
        score_col = score_columns.get(model)
        if not score_col or score_col not in df.columns:
            continue
        result, pred, matches = evaluate_model(
            df, gt, model, score_col, float(row.threshold), gap_frames,
            tolerance, args.fps, total_hours)
        results.append(result)
        pred.to_csv(out / f"{model}_predicted_episodes.csv", index=False)
        matches.to_csv(out / f"{model}_matched_episodes.csv", index=False)
        print(result)

    gt.to_csv(out / "ground_truth_window_episodes.csv", index=False)
    episode_results = pd.DataFrame(results)
    episode_results.to_csv(out / "episode_level_results.csv", index=False)

    # Paper-ready ablation table: window-level + episode-level metrics.
    paper = make_paper_table(ablation, episode_results)
    paper.to_csv(out / "paper_ablation_table.csv", index=False)
    save_latex_table(paper, out / "paper_ablation_table.tex")

    print("\n=== PAPER-READY ABLATION TABLE ===")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(paper.to_string(index=False))

    with (out / "episode_level_config.json").open("w", encoding="utf-8") as f:
        json.dump({**vars(args), "gap_frames": gap_frames,
                   "match_tolerance_frames": tolerance,
                   "observed_hours": total_hours,
                   "delay_definition": "first positive prediction-window episode"},
                  f, ensure_ascii=False, indent=2)
    print("Saved:", out.resolve())
    print("Paper CSV:", (out / "paper_ablation_table.csv").resolve())
    print("Paper LaTeX:", (out / "paper_ablation_table.tex").resolve())


if __name__ == "__main__":
    main()
