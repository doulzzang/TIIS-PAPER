# Canonical CVIU experiment mapping

The repository contains several development generations. This mapping marks
the files and result snapshots corresponding to the manuscript version.

| Manuscript component | Canonical source | Result snapshot |
|---|---|---|
| CHAD window preparation | `src/final/prepare_chad_windows.py` | `chad_windows_2s_4s_pairid.npz` (generated locally) |
| Five temporal baselines | `src/final/compare_modern_transformers.py` | `results/paper_final/modern_baseline_window_summary.csv` |
| Baseline episode protocol | `src/final/evaluate_modern_baseline_episodes.py` | `results/paper_final/modern_baseline_episode_summary.csv` |
| Agentic Memory tuning | `src/final/evaluate_agentic_memory.py` | `chad_agentic/results_episode_tuned_balanced/` |
| One-to-one episode matching and ablation | `src/final/make_paper_ablation.py` | `results/paper_final/paper_ablation_table.csv` and `.tex` |
| Hybrid diagnostic | `src/final/evaluate_hybrid_fusion.py` | `results/paper_final/hybrid_summary.csv` |

The manuscript-reported test set contains 299 ground-truth episodes. The
agentic ablation uses a 1.05-second episode gap and zero temporal match
tolerance. Thresholds and state parameters are selected on validation data;
the test set is used only for final reporting.

The DOCX manuscript is intentionally outside the public repository. CHAD raw
metadata and the full video collection are also not redistributed; see
`README.md` for the official dataset source and citation.
