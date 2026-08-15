"""GNN link-prediction implementation for the Kiva project.

This version intentionally uses only:
  1. loan nodes and partner nodes;
  2. observed loan-partner edges;
  3. the joined numeric and categorical loan features;
  4. two graph-convolution layers and a dot-product decoder.

It implements the core link-prediction task. Fairness is a separate later
evaluation layer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
from torch import nn


def number(value: str) -> float:
    """Convert a missing or malformed numeric field to zero."""
    # CSV values arrive as strings, so convert them before building tensors.
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class SimpleGCN(nn.Module):
    """Two graph-convolution layers followed by link scoring."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        # First layer learns a representation from loan/partner features.
        self.first = nn.Linear(input_size, hidden_size)
        # Second layer turns that representation into the final embedding.
        self.second = nn.Linear(hidden_size, output_size)

    def average_neighbors(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """Average neighbor features for every node."""
        # Each edge is stored as a source node and a destination node.
        source, target = edges
        # Start with an empty aggregate for every destination node.
        result = torch.zeros_like(x)
        # Add each source feature to the feature total of its destination.
        result.index_add_(0, target, x[source])
        # Count how many neighbors contributed to each destination.
        degree = torch.zeros(x.size(0))
        degree.index_add_(0, target, torch.ones(target.size(0)))
        # Divide by degree to obtain the mean neighbor feature.
        return result / degree.clamp_min(1).unsqueeze(1)

    def forward(self, features: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        # Each layer combines a node's own features with its neighbors' mean.
        # First message-passing step: collect information from neighbors.
        first_neighbors = self.average_neighbors(features, edges)
        # Combine each node's own features with its neighborhood information.
        hidden = torch.relu(self.first(features) + self.first(first_neighbors))
        # Repeat message passing at the hidden representation level.
        second_neighbors = self.average_neighbors(hidden, edges)
        # Produce one embedding vector per node.
        output = self.second(hidden) + self.second(second_neighbors)
        # Normalize embeddings so dot products are comparable.
        return nn.functional.normalize(output, dim=1)


def main() -> None:
    # Command-line options make the experiment easy to repeat.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=Path("data/processed/loans_joined.csv"))
    parser.add_argument("--results-file", type=Path, default=Path("outputs/evaluation/gnn_results.json"))
    parser.add_argument("--model-file", type=Path, default=Path("outputs/models/gnn_model.pt"),
                        help="Checkpoint used by the separate evaluation script")
    parser.add_argument("--max-loans", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=50413)
    # Read the values supplied by the user.
    args = parser.parse_args()
    # Make sampling and model initialization repeatable.
    random.seed(args.seed); torch.manual_seed(args.seed)

    # Use all usable joined features. IDs, free text, and field_partner_name
    # are excluded because they are either too sparse or reveal the target.
    numeric_fields = ["loan_amount", "funded_amount", "term_in_months", "lender_count", "mpi",
                      "rural_pct", "population_density", "sex_ratio", "gdp_per_capita",
                      "gdp_growth_rate", "agriculture_gva", "services_gva",
                      "internet_use", "urban_population"]
    categorical_fields = ["country", "country_code", "loan_region", "activity", "sector",
                          "currency", "repayment_interval", "borrower_gender", "theme_type",
                          "theme_country", "theme_region", "theme_iso", "mpi_match_type",
                          "country_profile_region"]
    # Each dictionary below represents one loan and its observed partner.
    rows = []
    with args.data_file.open(encoding="utf-8", errors="replace", newline="") as file:
        # DictReader lets us refer to columns by their names.
        for row in csv.DictReader(file):
            if not row.get("loan_id") or not row.get("partner_id") or not row.get("date"):
                continue
            # Keep the target IDs and all selected input features together.
            rows.append({"loan": row["loan_id"], "partner": row["partner_id"],
                         "date": row["date"],
                         # Keep protected attributes in the checkpoint so the
                         # separate fairness script can evaluate them later.
                         "gender": row.get("borrower_gender", "unknown").strip().lower() or "unknown",
                         "region": row.get("loan_region", "unknown").strip().lower() or "unknown",
                         "country": row.get("country", "unknown").strip().lower() or "unknown",
                         "partner_name": row.get("field_partner_name", "unknown").strip() or "unknown",
                         "numeric": [number(row.get(name, "")) for name in numeric_fields],
                         "categorical": [row.get(name, "unknown").strip().lower() or "unknown"
                                         for name in categorical_fields]})
    # Remove accidental duplicate loan IDs.
    rows = list({row["loan"]: row for row in rows}.values())
    if args.max_loans and len(rows) > args.max_loans:
        rows = random.sample(rows, args.max_loans)
    # Sorting by date makes the split simulate future prediction.
    rows.sort(key=lambda row: (row["date"], row["loan"]))
    print(f"loaded {len(rows):,} loans")

    # Temporal split: the model learns earlier links and predicts later links.
    # Use 70% for training, 15% for validation, and 15% for testing.
    train_end = int(len(rows) * .70)
    validation_end = int(len(rows) * .85)
    train_rows = rows[:train_end]
    test_rows = rows[validation_end:]

    # Assign one integer node ID to every loan and partner.
    # Give every loan a compact integer node ID for tensor indexing.
    loan_id = {row["loan"]: i for i, row in enumerate(rows)}
    # Partner IDs continue after the loan IDs in the same node index space.
    partner_id = {partner: len(loan_id) + i for i, partner in
                  enumerate(sorted({row["partner"] for row in rows}))}
    node_count = len(loan_id) + len(partner_id)

    # Standardize numeric features and one-hot encode categorical features.
    # Convert numeric columns from Python lists into a matrix.
    values = torch.tensor([row["numeric"] for row in rows], dtype=torch.float32)
    # Standardization prevents large dollar values from dominating small MPI values.
    values = (values - values.mean(0)) / values.std(0).clamp_min(1e-6)
    # Build a vocabulary for each categorical column.
    categories = []
    for column in range(len(categorical_fields)):
        categories.append(sorted({row["categorical"][column] for row in rows}))
    category_count = sum(len(column) for column in categories)
    # Add one extra feature to identify loan nodes from partner nodes.
    feature_size = len(numeric_fields) + category_count + 1
    features = torch.zeros(node_count, feature_size)
    features[:len(loan_id), :len(numeric_fields)] = values
    features[:len(loan_id), len(numeric_fields)] = 1.0  # loan node indicator
    offset = len(numeric_fields) + 1
    for column, vocabulary in enumerate(categories):
        positions = {value: i for i, value in enumerate(vocabulary)}
        for row_index, row in enumerate(rows):
            features[row_index, offset + positions[row["categorical"][column]]] = 1.0
        offset += len(vocabulary)

    # Build an undirected training graph. Test partner links are not included.
    # Store only historical target edges in the training graph.
    edge_pairs = []
    for row in train_rows:
        loan, partner = loan_id[row["loan"]], partner_id[row["partner"]]
        # Make the graph undirected so messages flow loan -> partner and back.
        edge_pairs.extend([(loan, partner), (partner, loan)])
    edges = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    model = SimpleGCN(feature_size, 32, 16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Positive examples are the observed historical loan-partner links.
    positive = [(loan_id[row["loan"]], partner_id[row["partner"]]) for row in train_rows]
    all_loans, all_partners = list(loan_id.values()), list(partner_id.values())
    known = set(positive)
    # Negative examples are randomly chosen loan-partner pairs not observed.
    negative = []
    while len(negative) < len(positive):
        pair = (random.choice(all_loans), random.choice(all_partners))
        if pair not in known:
            known.add(pair); negative.append(pair)

    # Train BCE on observed links (1) and sampled non-links (0).
    # Repeat the optimization step for the requested number of epochs.
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        # Compute one embedding for every loan and partner.
        embeddings = model(features, edges)
        pairs = positive + negative
        # A dot product is high when the two node embeddings point together.
        scores = torch.stack([(embeddings[a] * embeddings[b]).sum() for a, b in pairs])
        labels = torch.cat([torch.ones(len(positive)), torch.zeros(len(negative))])
        # BCE teaches positive pairs to score high and negative pairs low.
        loss = nn.functional.binary_cross_entropy_with_logits(scores, labels)
        loss.backward(); optimizer.step()
        print(f"epoch={epoch:03d} loss={loss.item():.4f}")

    # Evaluate each test partner against future loans only.
    # Switch off training-specific behavior before scoring test links.
    model.eval()
    with torch.no_grad(): embeddings = model(features, edges)

    # Save everything needed by the separate evaluation script. Keeping the
    # split rows and node mappings in the checkpoint makes evaluation repeatable.
    args.model_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "embeddings": embeddings.cpu(),
                "loan_id": loan_id, "partner_id": partner_id,
                "train_rows": train_rows, "validation_rows": rows[train_end:validation_end],
                "test_rows": test_rows, "max_loans": args.max_loans, "epochs": args.epochs,
                "seed": args.seed},
               args.model_file)
    print(f"checkpoint written to: {args.model_file}")
    # Evaluation is intentionally handled by evaluate_gnn.py. Returning here
    # prevents the old partner->loan diagnostic block below from being used as
    # an authoritative result file.
    return
    auc_values = []
    ndcg_values = {k: [] for k in (5, 10, 15, 20)}
    # Group future positive links by partner for ranking evaluation.
    test_by_partner = {}
    for row in test_rows:
        test_by_partner.setdefault(row["partner"], []).append(row["loan"])
    future_loans = [row["loan"] for row in test_rows]
    for partner, positive_loans in test_by_partner.items():
        # Candidate negatives come from future loans, not training loans.
        negatives = [loan for loan in future_loans if loan not in positive_loans]
        negatives = random.sample(negatives, min(len(negatives), max(50, len(positive_loans) * 10)))
        candidates = [(loan, 1) for loan in positive_loans] + [(loan, 0) for loan in negatives]
        random.shuffle(candidates)
        scores = [(embeddings[loan_id[loan]] * embeddings[partner_id[partner]]).sum().item()
                  for loan, _ in candidates]
        labels = [label for _, label in candidates]
        pos_scores = [s for s, y in zip(scores, labels) if y]
        neg_scores = [s for s, y in zip(scores, labels) if not y]
        # AUC measures how often a positive receives a higher score than a negative.
        auc_values.append(sum(float(p > n) + .5 * float(p == n) for p in pos_scores for n in neg_scores) /
                          max(1, len(pos_scores) * len(neg_scores)))
        # Sort candidates once, then evaluate the ranked list at several cutoffs.
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        for k in ndcg_values:
            order = ranked[:k]
            dcg = sum(labels[i] / math.log2(j + 2) for j, i in enumerate(order))
            ideal = sum(1 / math.log2(j + 2) for j in range(min(k, len(pos_scores)))) or 1
            ndcg_values[k].append(dcg / ideal)

    # Store the headline result in a small JSON file.
    result = {"loans": len(rows), "nodes": node_count, "train_edges": len(train_rows),
              "validation_loans": validation_end - train_end, "test_loans": len(test_rows),
              "test_auc": sum(auc_values) / max(1, len(auc_values)),
              "test_ndcg_at_5": sum(ndcg_values[5]) / max(1, len(ndcg_values[5])),
              "test_ndcg_at_10": sum(ndcg_values[10]) / max(1, len(ndcg_values[10])),
              "test_ndcg_at_15": sum(ndcg_values[15]) / max(1, len(ndcg_values[15])),
              "test_ndcg_at_20": sum(ndcg_values[20]) / max(1, len(ndcg_values[20])),
              "epochs": args.epochs}
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    args.results_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"NDCG@5={result['test_ndcg_at_5']:.4f} | "
          f"NDCG@10={result['test_ndcg_at_10']:.4f} | "
          f"NDCG@15={result['test_ndcg_at_15']:.4f} | "
          f"NDCG@20={result['test_ndcg_at_20']:.4f}")


if __name__ == "__main__":
    main()
