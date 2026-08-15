# Fairness-Aware Micro-Lending Partner Recommendation

**Group 13 | Final writeup draft**

> The ranking values below are from the full 50-epoch run. Runtime and the
> comparison row still need to be added.

## Abstract

We build a graph neural network that recommends Kiva field partners for a new
micro-loan application. The system combines loan transactions with partner,
regional, MPI, and country-level economic data. A temporal split tests whether
historical interactions can predict future loan-partner links. The final
pipeline separates model ranking quality from fairness auditing so that model
quality can be understood before a policy guardrail is applied.

The full 50-epoch run shows strong link ranking (AUC `0.9850`, NDCG@5
`0.8790`) above the original ranking target. Gender exposure is evaluated per
partner rather than with one misleading global maximum. Region is not used as
a protected fairness attribute because each partner operates in a specific
geography, making regional differences partly a partner-eligibility effect.

## 1. Problem and intended use

Given a new loan application, rank field partners that historically funded
similar loans. The output is a ranked list of partner IDs and names:

`loan_id -> [partner_1, partner_2, ...]`

This is a link-prediction problem, not a loan-success or repayment prediction
problem. A positive label means that the loan-partner link was observed in the
historical data. The system does not claim that a loan will be repaid.

## 2. Data and preprocessing

The pipeline reads five public sources: Kiva loan transactions, loan-theme IDs,
loan themes by region, MPI regional locations, and country-profile variables.
The sources are joined through loan/theme identifiers and standardized country
and region keys. Exact MPI matches are retained where available; otherwise the
pipeline records a country-level proxy and keeps the match type for auditing.

The processed table contains loan amount, funded amount, term, lender count,
sector, activity, currency, repayment interval, borrower gender, partner,
country, detailed region, MPI, rural percentage, population density, GDP, GDP
growth, agriculture/services share, internet use, and urban population.

**Data summary.** Processed loans: `657,698`; exact MPI match rate: `9.61%`;
country-proxy MPI rate: `73.29%`; unmatched MPI rate: `17.10%`.

## 3. Graph representation and objective

Each loan and partner is a node. An observed historical loan-partner
relationship is an undirected message-passing edge during training. Loan nodes
start with standardized numeric features and one-hot categorical features.
Partner nodes do not have loan-specific input features; they start with a
zero-initialized feature vector and learn a representation by averaging the
features of their connected loan neighbors. The graph is used to learn
embeddings for both node types.

At each GNN layer, every node combines its own representation with the mean of
all its training-graph neighbors. The neighbor count is determined by the
observed graph degree, not by a fixed manually selected number. Two layers
allow information to travel through approximately two hops, such as
loan -> partner -> another loan.

The GNN uses two readable mean-neighbor aggregation layers. For a loan-partner
pair `(l,p)`, the decoder is a dot product:

`score(l,p) = z_l dot z_p`

Training uses binary cross-entropy with observed links as positive examples and
sampled unobserved loan-partner pairs as negative examples:

`BCE = -[y log(sigmoid(score)) + (1-y) log(1-sigmoid(score))]`

The inference direction is loan-to-partner, matching the intended application
workflow.

## 4. Experimental design

Rows are sorted by date and split chronologically into 70% training, 15%
validation, and 15% test data. Historical training edges are included in the
message-passing graph; future test links are held out from training. The final
run used `657,698` loans, `50` epochs, and seed `50413`.

## 5. Model evaluation

We report AUC, average precision, Hits@5, and NDCG at 5, 10, 15, and 20. NDCG
is the primary ranking metric because a lender is most likely to inspect the
first few recommendations. The original proposal target was NDCG@10 or NDCG@5
at least 0.80; the final report will state the selected cutoff consistently.

### Full-run results

| Metric | Value | Target |
|---|---:|---:|
| AUC | 0.9850 | above random |
| Average precision | 0.8590 | report |
| Hits@5 | 0.9519 | report |
| NDCG@5 | 0.8790 | >= 0.80 |
| NDCG@10 | 0.8877 | report |
| NDCG@20 | 0.8924 | report |

These values are from the full 50-epoch run; the one-epoch run was used only
to validate the execution flow.

## 6. Gender fairness evaluation

For every partner, we calculate the selection rate for each borrower-gender
group:

`selection_rate(group, partner) = selected loans from group / test loans in group`

The partner-specific gap is the difference between the highest and lowest
group selection rates for that partner. The JSON output retains partner ID,
partner name, group counts, group rates, gap, and threshold pass/fail. We report
the average and exposure-weighted gap as summaries, while the per-partner table
is the primary evidence.

The full-run gender results, excluding the 81 unknown-gender queries from the
protected-group comparison, are: weighted gap `4.39 percentage points`, average
partner gap `2.42 percentage points`, and `85.7%` of compared partners meeting
the 5% target (174 of 203 partners). Unknown-gender records are reported separately rather than
treated as a third protected group. The region attribute is intentionally excluded from the fairness claim because
partner geography and regional availability are structurally linked.

## 7. Inference-time gender guardrail

The guardrail is applied after GNN scoring and does not change training weights
or embeddings. It processes female and male loan queries in alternating order.
When selecting each partner for a loan's top-10 slate, it applies a penalty to
partners already over-exposed for that query's gender relative to the other
gender. The final output always contains at least 10 candidates when the
candidate pool permits it.

The evaluation reports raw-versus-guarded fairness and ranking results. Because
candidate availability can prevent exact parity, the guardrail audit remains
the source of truth; it must not be described as a guaranteed zero-violation
policy until the full guarded run confirms that result.

### Guardrail example and penalty

Suppose a female loan has raw partner scores `A=0.95`, `B=0.90`, `C=0.85`,
and `D=0.80`. If partner A is currently selected at 40% for female loans but
20% for male loans, its exposure difference is `0.20`. With the default
guardrail penalty of `1.0`, the adjusted A score is:

`adjusted_score(A) = 0.95 - (1.0 x 0.20) = 0.75`

The guardrail can therefore select `B, C, D` instead of `A, B, C`. For a male
loan, A may receive no penalty if it is under-exposed for male queries. The
penalty is not learned by the GNN; it is a configurable inference parameter
(`--guardrail-penalty 1.0`). It should be tuned on validation data by comparing
ranking loss against the gender-gap target, then held fixed for the test run.

Guarded full-run values: NDCG@5 `0.8804` for the raw GNN score ranking,
weighted gender gap `3.89 percentage points`, and partner-level violation rate
`12.3%`. The guardrail improves exposure balance and raises the partner pass
rate from `85.7%` to `87.7%`, but it does not yet meet the strict 0% violation
target for every partner.

## 8. Team model and evaluation comparison

Checkpoint 2 divided the project into two two-person teams. Each team owns both
one model proposal and its corresponding evaluation proposal:

- **Team A:** hybrid gradient-boosted ranking with a deterministic post-filter;
  evaluation of ranking quality, selection-rate difference, and guardrail
  adherence.
- **Team B:** GraphSAGE/link prediction with an agentic fairness guardrail;
  evaluation of NDCG ranking quality, gender fairness, guardrail compliance,
  and scalability.

Our implementation is Team B's GraphSAGE track. The final report will include
both teams' results using the same test split where possible:

| Approach | AUC | NDCG@5 | NDCG@10 | Gender gap | Runtime |
|---|---:|---:|---:|---:|---:|
| GraphSAGE (this subgroup) | `[FULL-RUN VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` |
| Other team approach | `[OBTAIN TEAM RESULT]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` |

Each team should provide its own model and evaluation evidence; neither team is
expected to reimplement the other's approach.

## 9. Scalability and deployment discussion

The deployment path is: preprocess a new loan, build its feature vector, score
candidate partners from frozen embeddings, return the top 10 partner IDs/names,
then run the gender audit/guardrail. The final run must record training time,
evaluation time, inference time per loan, throughput, and memory usage. The
main deployment risks are cold-start loans/partners, changing partner coverage,
missing MPI values, historical popularity bias, and fairness trade-offs that
may reduce ranking quality.

## 10. Lessons learned and peer-review integration

The implementation became simpler and more auditable after separating
preprocessing, training, model evaluation, and fairness evaluation. A major
lesson was that a high AUC can coexist with weak top-k NDCG. Another was that
regional fairness is not interpretable when partner eligibility is geographic,
so the final fairness claim is limited to gender.

**Peer-review feedback integration:** `[ADD THE SPECIFIC FEEDBACK AND EXPLAIN
THE RESULTING CHANGE HERE]`.

## 11. Contributions and reproducibility

Team roles from Checkpoint 2 are:

- **Team A - Atanu Choudhury:** modeling proposal (hybrid GBDT ranking).
- **Team A - Rajesh Mattaparthi:** evaluation proposal.
- **Team A - Atanu Choudhury and Rajesh Mattaparthi:** implementation/testing.
- **Team B - Prateek Grover:** modeling proposal (GraphSAGE/link prediction).
- **Team B - Aditya Jaini:** evaluation proposal.
- **Team B - Prateek Grover and Aditya Jaini:** implementation/testing.

The final report should add any refinements to these responsibilities and
describe the shared preprocessing/presentation work.

Code repository: `[ADD PUBLIC REPOSITORY LINK]`  
Presentation PDF: `[ADD FINAL SLIDE LINK/FILE]`  
Presentation video: `[ADD VIDEO LINK]`

## 12. Final conclusion

The project delivers an interpretable loan-to-partner GNN pipeline with
reproducible data joins, temporal evaluation, ranking metrics, and partner-level
gender auditing. The full submission will make a clear distinction between
what is complete (core GNN and measurement pipeline) and what remains (second
model/baseline, active guardrail, full-run numbers, and scalability evidence).
