# Fairness-Aware Micro-Lending Partner Recommendation

**Group 13 | Final writeup draft**

> Numeric results in this draft are provisional one-epoch results. Replace
> every value marked `[FULL-RUN VALUE]` after the final experiment.

## Abstract

We build a graph neural network that recommends Kiva field partners for a new
micro-loan application. The system combines loan transactions with partner,
regional, MPI, and country-level economic data. A temporal split tests whether
historical interactions can predict future loan-partner links. The final
pipeline separates model ranking quality from fairness auditing so that model
quality can be understood before a policy guardrail is applied.

Our current one-epoch run shows that the GNN learns meaningful link structure
(AUC `[0.9005; replace with full-run value]`), but top-of-list ranking remains
below the original NDCG target (`NDCG@5 [0.3874]` versus target `0.80`). Gender
exposure is evaluated per partner rather than with one misleading global
maximum. Region is not used as a protected fairness attribute because each
partner operates in a specific geography, making regional differences partly a
partner-eligibility effect.

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

**Data summary.** Processed loans: `[FULL-RUN VALUE]`; exact MPI match rate:
`[FULL-RUN VALUE]`; country-proxy MPI rate: `[FULL-RUN VALUE]`; unmatched MPI
rate: `[FULL-RUN VALUE]`.

## 3. Graph representation and objective

Each loan and partner is a node. An observed historical loan-partner
relationship is an undirected message-passing edge during training. Loan nodes
carry standardized numeric features and one-hot categorical features; partner
nodes receive the graph's partner representation. The graph is used to learn
embeddings for both node types.

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
run will use `[FULL-RUN LOAN COUNT]` loans, `[FULL-RUN EPOCH COUNT]` epochs, and
seed `[FULL-RUN SEED]`.

## 5. Model evaluation

We report AUC, average precision, Hits@5, and NDCG at 5, 10, 15, and 20. NDCG
is the primary ranking metric because a lender is most likely to inspect the
first few recommendations. The original proposal target was NDCG@10 or NDCG@5
at least 0.80; the final report will state the selected cutoff consistently.

### Provisional one-epoch results

| Metric | Value | Target |
|---|---:|---:|
| AUC | 0.9005 | above random |
| Average precision | 0.3482 | report |
| Hits@5 | 0.6349 | report |
| NDCG@5 | 0.3874 | >= 0.80 |
| NDCG@10 | 0.4630 | report |
| NDCG@20 | 0.4887 | report |

These values are a baseline for the full run, not the final claim.

## 6. Gender fairness evaluation

For every partner, we calculate the selection rate for each borrower-gender
group:

`selection_rate(group, partner) = selected loans from group / test loans in group`

The partner-specific gap is the difference between the highest and lowest
group selection rates for that partner. The JSON output retains partner ID,
partner name, group counts, group rates, gap, and threshold pass/fail. We report
the average and exposure-weighted gap as summaries, while the per-partner table
is the primary evidence.

The current one-epoch gender results are provisional: weighted gap `[0.1654]`,
average partner gap `[0.0846]`, and `[85/153]` partners meeting the 5% target.
The region attribute is intentionally excluded from the fairness claim because
partner geography and regional availability are structurally linked.

## 7. Guardrail status and planned extension

The current system audits gender exposure but does not yet alter model scores.
Therefore guardrail violation rate is not reported as zero; it is marked
`measurement_only_not_enforced`. The next implementation step is a transparent
post-ranking policy that starts from at least 10 recommendations, checks the
batch-level gender criterion, and swaps candidates only when the documented
criterion is violated. We will report both recommendation quality before and
after this policy.

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
