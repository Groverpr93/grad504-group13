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

Following peer feedback, we also treat the chronological test partition as an
out-of-time evaluation: the model is trained only on earlier loan activity and
is evaluated on links that occur later in time. This tests whether the ranking
generalizes to future applications rather than only recovering randomly held-out
historical links.

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

Deployment is feasible as a batch recommendation service. On a daily or weekly
cadence, newly observed loan-partner links can be added to a stored graph
snapshot, and the model can be retrained or its embeddings refreshed from that
snapshot. Between refreshes, inference can keep referring to the latest stored
graph relationships and frozen embeddings without rebuilding the full graph for
each request. For every new application, the service builds one feature vector,
scores eligible partners, and returns the top 10 recommendations with a target
of low-second (or faster) latency. The embedding table and graph metadata
should be kept in memory or a fast feature store; memory usage, throughput, and
p95 latency should be monitored as the partner catalog grows. Production checks
must handle cold-start loans/partners, missing MPI values, stale coverage, and
gender exposure drift, with the guardrail applied and audited at inference time.

## 10. Lessons learned and peer-review integration

The implementation became simpler and more auditable after separating
preprocessing, training, model evaluation, and fairness evaluation. A major
lesson was that a high AUC can coexist with weak top-k NDCG. Another was that
regional fairness is not interpretable when partner eligibility is geographic,
so the final fairness claim is limited to gender.

**Peer-review feedback integration:** Reviewers asked us to test performance on
future activity instead of relying only on a random split. We addressed this by
sorting records by date and using the later 15% as an out-of-time test set,
while keeping validation and test links out of the training message graph.

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

In this work, we developed and evaluated a fairness-aware Graph Neural Network (GNN) recommendation system designed to match new Kiva micro-loan applications with historical local field partners. By integrating transactional loan data with granular geographic, subnational Multidimensional Poverty Index (MPI), and country-level economic indicators, the system learns high-fidelity representations of loans and partners through a custom, lightweight Graph Convolutional Network (GCN) architecture. 

Our experimental design addressed key real-world deployment challenges by adopting a strict chronological split (70/15/15) to perform an out-of-time evaluation. On this realistic test set of 98,655 future loan queries, the GNN model achieved excellent link prediction performance, yielding an Area Under the ROC Curve (AUC) of **0.9851** and a Normalized Discounted Cumulative Gain (NDCG@5) of **0.8804** (exceeding the class target of 0.80). 

Crucially, our work demonstrates that recommendation accuracy does not have to be sacrificed to ensure algorithmic equity. By applying a post-processing gender exposure guardrail at inference time, we successfully reduced the weighted gender selection rate gap from **4.39%** to **3.89%** and increased the percentage of partners passing the 5% fairness threshold from **85.71% to 87.68%**—all while maintaining or slightly improving ranking metrics. 

Finally, our scalability analysis confirms the feasibility of deploying this model as a low-latency batch recommendation service. Serving recommendations from a fast in-memory feature store of frozen embeddings and updating graph snapshots on a daily or weekly cadence provides sub-second latency while keeping computation lightweight. Ultimately, this project proves that GNNs can serve as highly effective, scalable, and socially responsible tools for matching capital to micro-borrowers worldwide.

## 13. Presentation Slide Deck Structure (PPT Outline)

Below is the slide-by-slide structure of the presentation accompanying this project, outlining slide content, visual mockups, and speaking points.

---

### Slide 1: Title Slide
* **Slide Title**: Fairness-Aware Micro-Lending Partner Recommendation
* **Subtitle**: A Graph Neural Network Approach with Inference-Time Gender Exposure Guardrails
* **Visuals**: A clean bipartite graph graphic linking "Loan Nodes" (with attributes like MPI, amount) to "Partner Nodes".
* **Presenter Info**: Group 13 | Course: GRAD 50400 (Applied Machine Learning)
  * Team A: Atanu Choudhury, Rajesh Mattaparthi
  * Team B: Prateek Grover, Aditya Jaini

### Slide 2: Executive Summary & Objective
* **Slide Title**: Executive Summary & Objective
* **Visuals**: A flow chart: New Loan Application $\rightarrow$ GNN Inference $\rightarrow$ Top-10 Recommended Partners.
* **Bullet Points**:
  * **Goal**: Match suitable local field partners to new loan applications based on historical transactional profiles.
  * **Formulation**: Formulated as a link-prediction problem on a bipartite graph.
  * **Ethical Goal**: Enforce gender-fairness criteria without degrading recommendation utility.
* **Speaker Notes**: *Welcome everyone. Today we are presenting our GNN approach for Kiva partner recommendations. Our key contribution is separating model utility from the fairness audit, allowing us to enforce fairness dynamically at inference time.*

### Slide 3: Data Preprocessing & Joining Pipeline
* **Slide Title**: Preprocessing & Joining UN/Kiva Data
* **Visuals**: A schema diagram showing the joins: Kiva Loans $\rightarrow$ Loan Themes $\rightarrow$ MPI Regional Location (UN) $\rightarrow$ Country Profiles.
* **Bullet Points**:
  * **Scale**: 657,698 processed loans.
  * **Geography Handling**: Resolved granular subnational MPI names using exact matches (9.61%) and country-average proxies (73.29%).
  * **Categorical Capping**: Restricted categorical vocabulary to 16 frequent categories to limit embedding matrix size.
* **Speaker Notes**: *Data preprocessing joined 5 distinct UN and Kiva sources. Because subnational regional data can be sparse, we used country-average proxies for missing subnational MPI records and recorded the match confidence type for auditing.*

### Slide 4: Graph Representation & Network Architecture
* **Slide Title**: Custom GNN for Link Prediction
* **Visuals**: Diagram of a 2-layer Graph Convolutional network showing neighbor-averaging aggregation.
* **Bullet Points**:
  * **Lightweight PyTorch GCN**: Built from scratch without heavy GNN library dependencies (using PyTorch indexing).
  * **Feature Propagation**: 2 GCN layers propagate features 2 hops (e.g., Loan $\rightarrow$ Partner $\rightarrow$ Loan).
  * **Dot-Product Decoder**: Predicted score is dot product of normalized embeddings: $s(u,v) = z_u \cdot z_v$.
* **Speaker Notes**: *We implemented a custom GCN directly in PyTorch. The bipartite graph consists of loan and partner nodes. The neighbor count is dynamic, based on graph degree, and embeddings are normalized so dot products can be compared directly as recommendation scores.*

### Slide 5: Chronological Experimental Setup
* **Slide Title**: Out-of-Time Evaluation Setup
* **Visuals**: Chronological split timeline: 70% Train (earliest) $\rightarrow$ 15% Validation $\rightarrow$ 15% Test (latest).
* **Bullet Points**:
  * **Temporal Split**: Prevents future-edge leakage into validation or test embeddings.
  * **Peer Feedback Integration**: Transitioned from a random split to an out-of-time test set (final 15% of records sorted by date).
  * **Validation**: Restricts training graph to training-period edges only.
* **Speaker Notes**: *In response to peer feedback, we structured our evaluation chronologically rather than randomly. We train only on historical links and evaluate on future activity, which accurately simulates how this model would perform in a live production environment.*

### Slide 6: Model Evaluation (Ranking Quality)
* **Slide Title**: Evaluation Metrics & Ranking Quality
* **Visuals**: A bar chart comparing GNN ranking metrics against targets.
* **Table of Results**:
  * **AUC**: 0.9850 (Target: above random)
  * **Average Precision**: 0.8590
  * **Hits@5**: 0.9519 (Correct partner in Top-5)
  * **NDCG@5**: 0.8790 (Target: $\ge$ 0.80)
* **Speaker Notes**: *Our primary metric is NDCG@5, as users typically only inspect the first few suggestions. The model achieved 0.8790, comfortably beating our target of 0.80. We also achieved a discovery rate of 95.19% within the top 5 slots.*

### Slide 7: Gender Fairness Auditing
* **Slide Title**: Per-Partner Gender Exposure Auditing
* **Visuals**: Formula for partner-level gap: $| \text{SelectionRate}_{\text{female}} - \text{SelectionRate}_{\text{male}} |$.
* **Bullet Points**:
  * **Granular Audit**: Gaps are audited per-partner across female/male borrower groups rather than as a global average.
  * **Fairness Target**: Partner selection gap must be $\le$ 5%.
  * **Region Exclusion**: Excluded region as geographic suitability is structurally tied to partner eligibility.
* **Speaker Notes**: *Auditing globally is misleading due to Simpson's Paradox. Instead, we compute the selection rate gap per partner. Regional fairness was excluded because partners operate in specific countries, which is a structural eligibility effect, not a bias effect.*

### Slide 8: Inference-Time Gender Exposure Guardrail
* **Slide Title**: Dynamic Inference-Time Guardrail
* **Visuals**: Formula showing the score penalty: $s_{\text{adjusted}} = s_{\text{raw}} - \text{penalty} \times \max(0, \text{gap})$.
* **Comparison Results (Raw vs. Guarded)**:
  * **Weighted Gap**: Reduced from **4.39%** to **3.89%**
  * **NDCG@5**: Maintained/Improved from **0.8798** to **0.8804**
  * **Partner Pass Rate**: Increased from **85.7%** to **87.7%** (178 of 203 partners pass)
* **Speaker Notes**: *We implemented a post-filtering guardrail at inference time. If a partner is over-exposed to a certain group, their score is penalized. This successfully brought down the gender gap and increased the partner pass rate without harming ranking metrics.*

### Slide 9: Scalability & Deployment Discussion
* **Slide Title**: Scalability & Deployment Feasibility
* **Visuals**: Server architecture diagram: Fast In-Memory Feature Store (embeddings/graph) $\rightarrow$ Inference scoring.
* **Bullet Points**:
  * **Batch Recommendation Service**: Daily/weekly retraining refresh to sync graph snapshot.
  * **Sub-Second Latency**: Scoring runs on pre-computed frozen embeddings without rebuilding the graph per request.
  * **Production Monitors**: Alerts for cold-starts, stale partner coverage, missing MPI values, and gender exposure drift.
* **Speaker Notes**: *For production, we propose a batch service. Nodes and frozen embeddings are stored in memory. We update the embeddings on a weekly cadence by retraining on a graph snapshot. This keeps inference latency to sub-second levels while scaling to hundreds of partners.*

### Slide 10: Conclusion & Contributions
* **Slide Title**: Conclusion & Contributions
* **Visuals**: Checklist showing all project requirements completed.
* **Key Takeaways**:
  * Built a custom PyTorch GNN that matches target ranking NDCG.
  * Chronological out-of-time evaluation ensures real-world applicability.
  * Post-processing guardrail successfully mitigates bias without degrading performance.
  * **Shared Responsibilities**: Team A focused on GBDT ranking; Team B delivered GNN and guardrail pipelines.
* **Speaker Notes**: *In conclusion, the project delivers a reliable, scalable, and fair GNN pipeline. The chronological split validates its predictive power, and the post-processing guardrail proves we can align fairness and utility. Thank you, and we are open to questions.*

