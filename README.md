#  Agentic Memory for Actor-Centric Video Anomaly Detection

This repository contains the implementation, evaluation scripts, selected experiment videos, and result tables supporting the CVIU manuscript. The manuscript DOCX is not redistributed in this public repository.

The complete CHAD dataset is not redistributed here. Download it from the official [TeCSAR-UNCC/CHAD repository](https://github.com/TeCSAR-UNCC/CHAD) and follow its license and citation requirements. Download `CHAD_Meta` and the official split files before preparing windows.

`CHAD_Selected_Videos/` contains the 19 videos used for the actor-pair annotation and related experiments. They are tracked with Git LFS because several exceed GitHub's normal file limit. Do not add the complete CHAD video collection.

The original `CHAD_Meta/`, virtual environments, generated `.npz` features, and trained checkpoints are excluded from Git. Existing CSV, JSON, and LaTeX result tables document the reported experiments. The paper-ready table is `chad_agentic/results_agentic_ablation_paper/paper_ablation_table.tex`.

From the repository root, install dependencies with `pip install -r chad_comparison/requirements.txt`. The canonical commands and source files are under `src/final/`; the default protocol is a 2-second observation window followed by a 4-second evaluation interval. Use official CHAD train/validation/test splits and fit normalization and thresholds on training and validation data only.

The manuscript is submitted separately. Install Git LFS before downloading selected videos and run `git lfs pull`. The repository must preserve CHAD attribution and original terms.

## Canonical manuscript implementation

The final paper pipeline is collected under `src/final/`. Use
`FINAL_PIPELINE.md` for the mapping from each manuscript experiment to its
canonical source and result snapshot. Older versioned scripts remain in the
original directories as development history.
