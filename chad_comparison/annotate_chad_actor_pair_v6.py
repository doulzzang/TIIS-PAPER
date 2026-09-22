#!/usr/bin/env python3
"""Interactive actor/pair ground-truth annotation for selected CHAD videos.

Mouse: click a bounding box to toggle its person ID.
Keys:
  space pause/play, a/d -/+10 frames, j/l -/+30 frames
  s single_actor, p pair_interaction, g group_interaction, u unknown
  c clear selected IDs, Enter save segment and move next
  b previous segment, n next without saving, q quit
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import cv2
import numpy as np

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16)
]
FIELDS = ["video_id", "start_frame", "end_frame", "event_type",
          "abnormal_actor_ids", "abnormal_pair_ids", "annotator_note"]


def segments(labels):
    labels = np.asarray(labels).astype(bool)
    changes = np.diff(np.r_[False, labels, False].astype(np.int8))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def find_videos(root):
    extensions = {".mp4", ".avi", ".mov", ".mkv"}
    return sorted(p for p in Path(root).rglob("*")
                  if p.suffix.lower() in extensions)


def load_existing(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class Annotator:
    def __init__(self, video, annotation, labels, output, rows, args):
        self.video = Path(video)
        self.video_id = self.video.stem
        self.annotation = annotation
        self.labels = labels
        self.output = output
        self.rows = rows
        self.args = args
        self.parts = segments(labels)
        self.index = 0
        self.selected = set()
        self.event_type = "unknown"
        self.paused = False
        self.frame_no = 0
        self.scale = 1.0
        self.boxes = {}
        self.status_message = ""
        self.pending_auto_save = False
        self.auto_paused_for_segment = False
        parent_names = {p.lower() for p in self.video.parent.parts}
        self.folder_mode = (
            "single" if "single" in parent_names
            else "multi" if "multi" in parent_names
            else "manual"
        )
        self.window = "CHAD Actor-Pair Annotation"
        self.cap = cv2.VideoCapture(str(video))
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video: " + str(video))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or args.fps

    def existing_for_segment(self, start, end):
        for row in self.rows:
            if (row["video_id"] == self.video_id and
                    int(row["start_frame"]) == start and
                    int(row["end_frame"]) == end):
                ids = row.get("abnormal_actor_ids", "")
                self.selected = {x for x in ids.split(";") if x}
                self.event_type = row.get("event_type", "unknown")
                return
        self.selected, self.event_type = set(), "unknown"

    def goto_segment(self, index):
        self.index = max(0, min(index, len(self.parts) - 1))
        start, end = self.parts[self.index]
        self.frame_no = max(0, start - round(self.args.context_sec * self.fps))
        self.existing_for_segment(start, end)
        self.auto_paused_for_segment = False
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_no)

    def mouse(self, event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ox, oy = x / self.scale, y / self.scale
        candidates = []
        for pid, (bx, by, bw, bh) in self.boxes.items():
            if bx <= ox <= bx + bw and by <= oy <= by + bh:
                candidates.append((bw * bh, pid))
        if candidates:
            pid = min(candidates)[1]
            if pid in self.selected:
                self.selected.remove(pid)
                self.status_message = "ID " + pid + " deselected"
            else:
                self.selected.add(pid)
                self.status_message = "ID " + pid + " selected"
            self.paused = True
            if not self.args.manual_save:
                if self.folder_mode == "single" and len(self.selected) == 1:
                    self.event_type = "single_actor"
                    self.pending_auto_save = True
                    self.status_message = "Single actor selected: auto-saving"
                elif self.folder_mode == "multi" and len(self.selected) == 2:
                    self.event_type = "pair_interaction"
                    self.pending_auto_save = True
                    self.status_message = "Pair selected: auto-saving"

    def draw(self, frame):
        people = self.annotation.get(int(self.frame_no), {})
        self.boxes = {}
        for pid_raw, value in people.items():
            pid = str(pid_raw)
            bbox = np.nan_to_num(np.asarray(value[0], float))
            pose = np.nan_to_num(np.asarray(value[1], float))
            if bbox.shape != (4,):
                continue
            # Official CHAD annotation format:
            # bbox = [x, y, width, height], pose = [x, y, confidence].
            # Do not swap x/y here.  Swapping them displaces the overlay and
            # makes the person appear to be lying sideways.
            bx, by, bw, bh = bbox
            self.boxes[pid] = (bx, by, bw, bh)
            color = (0, 0, 255) if pid in self.selected else (0, 255, 0)
            cv2.rectangle(frame, (round(bx), round(by)),
                          (round(bx + bw), round(by + bh)), color, 3)
            cv2.putText(frame, "ID " + pid, (round(bx), max(25, round(by) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2)
            if pose.shape == (17, 3):
                # Some distributed CHAD metadata files store keypoint
                # coordinates as YXC although the README specifies XYC.
                # Select the orientation whose confident joints best fit the
                # corresponding, already-correct XYWH person box.
                confident = pose[:, 2] >= self.args.pose_conf
                if np.any(confident):
                    margin_x = max(3.0, 0.15 * bw)
                    margin_y = max(3.0, 0.15 * bh)

                    def inside_score(points):
                        px, py = points[:, 0], points[:, 1]
                        inside = ((px >= bx - margin_x) &
                                  (px <= bx + bw + margin_x) &
                                  (py >= by - margin_y) &
                                  (py <= by + bh + margin_y))
                        return int(np.count_nonzero(inside & confident))

                    swapped = pose.copy()
                    swapped[:, [0, 1]] = swapped[:, [1, 0]]
                    if inside_score(swapped) > inside_score(pose):
                        pose = swapped
                for a, b in SKELETON:
                    if pose[a, 2] >= self.args.pose_conf and pose[b, 2] >= self.args.pose_conf:
                        cv2.line(frame, tuple(np.round(pose[a, :2]).astype(int)),
                                 tuple(np.round(pose[b, :2]).astype(int)), color, 2)
        start, end = self.parts[self.index]
        anomaly = start <= self.frame_no <= end
        top_color = (0, 0, 255) if anomaly else (255, 255, 255)
        lines = [
            f"{self.video_id}  segment {self.index+1}/{len(self.parts)}  "
            f"frames {start}-{end}  current {self.frame_no}",
            "ANOMALY" if anomaly else "context",
            "selected IDs: " + (",".join(sorted(self.selected)) or "none"),
            "type: " + self.event_type + "  folder mode: " + self.folder_mode,
            "mouse=toggle ID | s=single p=pair g=group u=unknown",
            "space=play/pause a/d=10 j/l=30 c=clear b=prev n=skip q=quit"
        ]
        if self.status_message:
            lines.append("STATUS: " + self.status_message)
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (20, 35 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .68,
                        top_color if i < 2 else (0, 255, 255), 2)
        return frame

    def save_current(self):
        start, end = self.parts[self.index]
        ids = sorted(self.selected)
        pair = ""
        if self.event_type == "pair_interaction" and len(ids) == 2:
            pair = ids[0] + "-" + ids[1]
        row = {
            "video_id": self.video_id, "start_frame": start, "end_frame": end,
            "event_type": self.event_type,
            "abnormal_actor_ids": ";".join(ids),
            "abnormal_pair_ids": pair, "annotator_note": ""
        }
        self.rows = [r for r in self.rows if not (
            r["video_id"] == self.video_id and
            int(r["start_frame"]) == start and int(r["end_frame"]) == end)]
        self.rows.append(row)
        save_rows(self.output, self.rows)
        print("Saved", row)

    def selection_is_valid(self):
        count = len(self.selected)
        if self.event_type == "single_actor":
            return count == 1
        if self.event_type == "pair_interaction":
            return count == 2
        if self.event_type == "group_interaction":
            return count >= 2
        return count >= 1

    def validate_or_message(self):
        if not self.selected:
            self.status_message = "Select at least one actor ID"
            return False
        if self.event_type == "single_actor" and len(self.selected) != 1:
            self.status_message = "Single actor requires exactly one ID"
            return False
        if self.event_type == "pair_interaction" and len(self.selected) != 2:
            self.status_message = "Pair interaction requires exactly two IDs"
            return False
        if self.event_type == "group_interaction" and len(self.selected) < 2:
            self.status_message = "Group interaction requires two or more IDs"
            return False
        return True

    def save_and_advance(self):
        if not self.validate_or_message():
            self.paused = True
            return False
        self.save_current()
        self.status_message = "Saved"
        if self.index + 1 < len(self.parts):
            self.goto_segment(self.index + 1)
            return True
        return False

    def run(self):
        if not self.parts:
            print("No anomalous segment:", self.video_id)
            return self.rows
        # AUTOSIZE keeps mouse coordinates aligned with the displayed image.
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window, self.mouse)
        self.goto_segment(0)
        while True:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_no)
            ok, frame = self.cap.read()
            if not ok:
                break
            shown = self.draw(frame)
            h, w = shown.shape[:2]
            self.scale = min(1.0, self.args.display_width / w)
            if self.scale != 1.0:
                shown = cv2.resize(shown, None, fx=self.scale, fy=self.scale)
            cv2.imshow(self.window, shown)
            # Always poll so a mouse-only annotation can advance without
            # requiring a keyboard event while paused.
            key = cv2.waitKey(
                max(1, round(1000 / self.fps))) & 0xFF

            # Stop at a selectable frame inside the anomaly, rather than
            # waiting until the actors have already left the scene.
            start, end = self.parts[self.index]
            required_people = 2 if self.folder_mode == "multi" else 1
            if (not self.args.manual_save and not self.paused and
                    not self.auto_paused_for_segment and
                    start <= self.frame_no <= end and
                    len(self.boxes) >= required_people):
                self.paused = True
                self.auto_paused_for_segment = True
                self.status_message = (
                    f"Select {required_people} abnormal actor(s)"
                )
            if self.pending_auto_save:
                self.pending_auto_save = False
                if self.save_and_advance():
                    continue
                break
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                self.paused = not self.paused
            elif key == ord("a"):
                self.frame_no = max(0, self.frame_no - 10)
            elif key == ord("d"):
                self.frame_no += 10
            elif key == ord("j"):
                self.frame_no = max(0, self.frame_no - 30)
            elif key == ord("l"):
                self.frame_no += 30
            elif key == ord("c"):
                self.selected.clear()
                self.status_message = "Selection cleared"
            elif key == ord("s"):
                self.event_type = "single_actor"
                self.status_message = "Type: single_actor"
                if not self.args.manual_save and self.selection_is_valid():
                    if self.save_and_advance():
                        continue
                    break
            elif key == ord("p"):
                self.event_type = "pair_interaction"
                self.status_message = "Type: pair_interaction"
                if not self.args.manual_save and self.selection_is_valid():
                    if self.save_and_advance():
                        continue
                    break
            elif key == ord("g"):
                self.event_type = "group_interaction"
                self.status_message = "Type: group_interaction"
                if not self.args.manual_save and self.selection_is_valid():
                    if self.save_and_advance():
                        continue
                    break
            elif key == ord("u"):
                self.event_type = "unknown"
                self.status_message = "Type: unknown; press Enter to save"
            elif key == ord("b"):
                self.goto_segment(self.index - 1)
                continue
            elif key == ord("n"):
                if self.index + 1 < len(self.parts):
                    self.goto_segment(self.index + 1)
                    continue
                break
            elif key in (10, 13):
                if self.save_and_advance():
                    continue
                break
            if not self.paused:
                self.frame_no += 1
                _, end = self.parts[self.index]
                stop_frame = end + round(self.args.context_sec * self.fps)
                if self.frame_no >= stop_frame:
                    self.frame_no = stop_frame
                    self.paused = True
                    self.status_message = (
                        "End of anomaly context: select IDs and press s/p/g")
        self.cap.release()
        cv2.destroyWindow(self.window)
        return self.rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="../CHAD_Selected_Videos")
    p.add_argument("--meta-dir", default="../CHAD_Meta")
    p.add_argument("--output", default="actor_pair_ground_truth.csv")
    p.add_argument("--fps", type=float, default=30)
    p.add_argument("--context-sec", type=float, default=2)
    p.add_argument("--pose-conf", type=float, default=.20)
    p.add_argument("--display-width", type=int, default=1400)
    p.add_argument("--manual-save", action="store_true",
                   help="Require Enter after s/p/g; default saves automatically")
    args = p.parse_args()
    output = Path(args.output)
    rows = load_existing(output)
    videos = find_videos(args.videos)
    if not videos:
        raise FileNotFoundError("No videos under " + args.videos)
    for video in videos:
        name = video.stem
        ann_file = Path(args.meta_dir) / "annotations" / (name + ".pkl")
        label_file = Path(args.meta_dir) / "anomaly_labels" / (name + ".npy")
        if not ann_file.exists() or not label_file.exists():
            print("SKIP missing metadata:", name)
            continue
        with ann_file.open("rb") as f:
            annotation = pickle.load(f)
        labels = np.load(label_file, allow_pickle=False)
        rows = Annotator(video, annotation, labels, output, rows, args).run()
    print("Ground truth:", output.resolve())


if __name__ == "__main__":
    main()
