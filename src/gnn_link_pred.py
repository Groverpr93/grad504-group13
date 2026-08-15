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
import random
from collections import Counter
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
    parser.add_argument("--log-file", type=Path, default=Path("outputs/logs/gnn_training.jsonl"),
                        help="One JSON record per completed epoch")
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="Save a model-state checkpoint every N epochs")
    parser.add_argument("--model-file", type=Path, default=Path("outputs/models/gnn_model.pt"),
                        help="Checkpoint used by the separate evaluation script")
    parser.add_argument("--max-loans", type=int, default=50000)
    parser.add_argument("--max-categories", type=int, default=16,
                        help="Keep only the most frequent values per categorical field")
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
        # A dense one-hot matrix cannot hold tens of thousands of region names.
        # Keep frequent values and map the long tail to the existing unknown
        # bucket. This preserves all loans while keeping memory predictable.
        counts = Counter(row["categorical"][column] for row in rows)
        keep = {value for value, _ in counts.most_common(max(1, args.max_categories - 1))}
        keep.add("unknown")
        for row in rows:
            if row["categorical"][column] not in keep:
                row["categorical"][column] = "unknown"
        categories.append(sorted(keep))
    category_count = sum(len(column) for column in categories)
    # Add one extra feature to identify loan nodes from partner nodes.
    feature_size = len(numeric_fields) + category_count + 1
    estimated_gb = node_count * feature_size * 4 / (1024 ** 3)
    print(f"feature matrix: {node_count:,} x {feature_size} float32 "
          f"(about {estimated_gb:.2f} GB)")
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
    print(f"graph edges ready: {len(train_rows):,} historical links", flush=True)
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
        if len(negative) and len(negative) % 100000 == 0:
            print(f"negative pairs sampled: {len(negative):,}/{len(positive):,}", flush=True)

    # Convert pairs once so each epoch uses one vectorized tensor operation,
    # rather than creating nearly one million tiny Python-level dot products.
    pair_tensor = torch.tensor(positive + negative, dtype=torch.long)
    pair_sources = pair_tensor[:, 0]
    pair_targets = pair_tensor[:, 1]
    labels = torch.cat([torch.ones(len(positive)), torch.zeros(len(negative))])

    # Train BCE on observed links (1) and sampled non-links (0).
    # Repeat the optimization step for the requested number of epochs.
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.write_text("", encoding="utf-8")
    args.model_file.parent.mkdir(parents=True, exist_ok=True)
    setup_record = {"stage": "setup_complete", "loans": len(rows),
                    "nodes": node_count, "train_links": len(train_rows),
                    "feature_size": feature_size}
    args.log_file.write_text(json.dumps(setup_record) + "\n", encoding="utf-8")
    print("setup complete; starting epoch 1", flush=True)
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        # Compute one embedding for every loan and partner.
        embeddings = model(features, edges)
        # A dot product is high when the two node embeddings point together.
        # Indexing whole rows lets PyTorch calculate all pair scores together.
        scores = (embeddings[pair_sources] * embeddings[pair_targets]).sum(dim=1)
        # BCE teaches positive pairs to score high and negative pairs low.
        loss = nn.functional.binary_cross_entropy_with_logits(scores, labels)
        loss.backward(); optimizer.step()
        record = {"epoch": epoch, "loss": loss.item()}
        with args.log_file.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record) + "\n")
        print(f"epoch={epoch:03d} loss={loss.item():.4f} | log={args.log_file}")
        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            periodic_file = args.model_file.with_name(
                f"{args.model_file.stem}_epoch{epoch:03d}{args.model_file.suffix}")
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "loss": loss.item(), "seed": args.seed}, periodic_file)
            print(f"checkpoint written to: {periodic_file}")

    # Freeze the final embeddings for the separate evaluation script.
    model.eval()
    with torch.no_grad(): embeddings = model(features, edges)

    # Save everything needed by the separate evaluation script. Keeping the
    # split rows and node mappings in the checkpoint makes evaluation repeatable.
    torch.save({"model_state": model.state_dict(), "embeddings": embeddings.cpu(),
                "loan_id": loan_id, "partner_id": partner_id,
                "train_rows": train_rows, "validation_rows": rows[train_end:validation_end],
                "test_rows": test_rows, "max_loans": args.max_loans, "epochs": args.epochs,
                "seed": args.seed},
               args.model_file)
    print(f"checkpoint written to: {args.model_file}")


if __name__ == "__main__":
    main()
