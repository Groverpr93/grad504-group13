# Checkpoint 2 to final: status

The Checkpoint 2 design divided the class project into two two-person teams.
Each team owns one model proposal and its corresponding evaluation proposal.
Team A owns the hybrid gradient-boosted ranking/post-filter approach; Team B
owns the GraphSAGE/link-prediction and guardrail approach. The final report
should compare both teams' model-plus-evaluation results.

## Completed today

- Joined all five supplied Kiva/UN data sources into a reproducible processed table.
- Added MPI exact-match and country-proxy join reporting.
- Built a loan-partner graph with joined numeric and categorical loan features.
- Implemented a readable two-layer message-passing GNN and BCE link objective.
- Used a chronological 70/15/15 train/validation/test split.
- Corrected inference direction to new loan -> ranked partner recommendations.
- Separated model evaluation from training.
- Added AUC, average precision, Hits@5, and NDCG@5/@10/@15/@20.
- Added partner-level gender exposure analysis with per-partner group rates.
- Excluded region from the fairness benchmark because partner geography confounds regional comparisons.
- Added a notebook that visualizes the graph and link-prediction objective.

## Still pending

- Obtain the other team's final model/evaluation results and add a concise comparison table.
- Add an actual fairness-aware reranking/guardrail layer; current fairness output is measurement-only.
- Measure guardrail adherence/violation rate after the guardrail is active.
- Run the full selected epoch/data experiment and replace one-epoch placeholders.
- Report runtime, inference latency, throughput, and memory/scalability behavior.
- Add a random or popularity baseline so AUC/NDCG results have context.
- Document peer-review feedback and the resulting design change.
- Add final team roles/contributions, public code link, presentation PDF, and video link.
- Export the final writeup to PDF and complete the final presentation.
