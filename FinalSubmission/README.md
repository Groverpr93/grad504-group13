# Final submission working package

This folder is the working package for the final writeup and presentation. It
contains the current code, notebook, and one-epoch evaluation artifacts. The
numbers in the draft are marked provisional and should be replaced after the
full training run.

## Package contents

- `final_writeup_draft.md`: the narrative for the final report.
- `CHECKPOINT2_GAP_ANALYSIS.md`: requirements still pending from the design review.
- `code/`: preprocessing, GNN training, and separate evaluation scripts.
- `notebooks/`: graph and link-prediction visual explanation.
- `results/`: current one-epoch checkpoint and JSON outputs.

## Final run checklist

1. Run preprocessing and save the join report.
2. Train the GNN for the selected full-data epoch count.
3. Run `evaluate_gnn.py` and replace provisional result files.
4. Obtain the other team's model results and add the team-level comparison table.
5. Measure training/inference runtime and recommendation throughput.
6. Add peer-review feedback, team roles, repository link, and presentation/video links.
7. Export the completed narrative to PDF and prepare the final slide deck/video.
