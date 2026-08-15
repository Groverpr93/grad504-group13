# Code guide

Use the scripts in this order:

1. `preprocess_kiva.py` joins the five CSV inputs and writes `data/processed`.
2. `gnn_link_pred.py` reads `loans_joined.csv`, builds the loan-partner graph,
   trains the model and saves a checkpoint.
3. `evaluate_gnn.py` loads that checkpoint and ranks partners for each new
   loan. It writes model metrics, one
   fairness file per critical attribute, and a benchmark summary separately.

The script is intentionally self-contained so the graph, training loop,
objective function, and evaluation are visible in one place.

## One-epoch flow test

Run a small, repeatable test before a long experiment:

```bash
python src/gnn_link_pred.py --max-loans 50000 --epochs 1 \
  --model-file outputs/models/gnn_model_epoch1.pt \
  --results-file outputs/evaluation/gnn_results_epoch1.json

python src/evaluate_gnn.py \
  --model-file outputs/models/gnn_model_epoch1.pt \
  --model-results outputs/evaluation/model_evaluation_epoch1.json \
  --fairness-dir outputs/evaluation/fairness_epoch1 \
  --summary-file outputs/evaluation/benchmark_summary_epoch1.json
```

Inference returns at least 10 partner recommendations for each loan. Fairness
is evaluated across the batch of loan queries because each individual loan has
only one gender/region group:

```bash
python src/evaluate_gnn.py --model-file outputs/models/gnn_model_epoch1.pt \
  --minimum-k 10 --eval-batch-size 10 --variance-threshold 0.0025
```

The inference audit is saved in `outputs/evaluation/reranked_inference.json`.

The model file contains the frozen embeddings and temporal test split. The
evaluation script does not retrain the model. `fairness_gender.json` and
`fairness_region.json` are measurement-only outputs; the guardrail is reported
as not enforced until a fairness-aware training objective is added.

Inference output is keyed by loan ID and contains partner ID/name objects,
matching the intended production flow for a new loan application.

Fairness JSON reports a `per_partner` section for gender. Region is not used as
a fairness attribute because partners operate in specific geographies, so
regional differences mostly reflect partner eligibility. For every partner it lists each
group's selection rate, selected/query counts, that partner's gap, and whether
the 5% threshold is met. Aggregate average and weighted gaps are included only
as context; the partner threshold pass rate is the main summary.
Partner entries include both `partner_id` and the Kiva `field_partner_name`.
