# Final manuscript pipeline

These files are the only canonical versions used to reproduce the CVIU
manuscript results. Development-version scripts are not included in the
public source tree.

Run commands from the repository root with `PYTHONPATH=src/final`.

1. Prepare the official CHAD metadata into a pair-aware window file with
   `prepare_chad_windows.py`.
2. Run `compare_modern_transformers.py` with models `tcn transformer
   actor_pair_transformer patchtst itransformer`, seeds `41 42 43`, and the
   same prepared windows.
3. Run `evaluate_agentic_memory.py` on the pair-aware windows. Its selection
   is validation-episode based and its online memory does not inspect labels.
4. Run `make_paper_ablation.py` on the selected test score and ablation CSVs.
5. Run `evaluate_modern_baseline_episodes.py` on the five baseline score files.

The committed `results/paper_final/` files are the result snapshots used for
the manuscript tables. The source scripts do not require the raw RGB videos
for model evaluation; the selected videos are retained for actor-pair
annotation and supporting experiments.
