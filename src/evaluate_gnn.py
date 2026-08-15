"""Evaluate a saved GNN checkpoint in two small, separate sections.

The model section measures link-ranking quality.  The fairness section measures
whether the top recommendations are selected at similar rates for each
protected attribute.  Keeping these outputs separate makes it easy to inspect
the core model before adding a fairness-aware training objective.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch


def row_value(row, attribute):
    """Read an attribute from a current or older checkpoint row."""
    # New checkpoints store attributes by name.  The fallback keeps old files
    # usable (categorical order is defined in gnn_link_pred.py).
    if isinstance(row, dict) and attribute in row:
        return str(row[attribute] or "unknown").strip().lower()
    fallback = {"gender": 7, "region": 4, "country": 0}
    values = row.get("categorical", []) if isinstance(row, dict) else []
    position = fallback.get(attribute)
    return str(values[position] if position is not None and position < len(values) else "unknown")


def candidates_by_loan(test_rows, seed):
    """Build future partner candidates for each incoming loan."""
    random.seed(seed)
    all_future = list({row["partner"] for row in test_rows})
    positives = defaultdict(set)
    for row in test_rows:
        positives[row["loan"]].add(row["partner"])
    output = {}
    for loan, positive_partners in positives.items():
        # Negatives are other future partners, so no training edge leaks into the
        # ranking task.  Ten negatives per positive keeps the run practical.
        pool = [partner for partner in all_future if partner not in positive_partners]
        count = min(len(pool), max(50, len(positive_partners) * 10))
        negatives = random.sample(pool, count)
        pairs = [(partner, 1) for partner in positive_partners] + [(partner, 0) for partner in negatives]
        random.shuffle(pairs)
        output[loan] = pairs
    return output


def score_candidates(checkpoint, grouped):
    """Return scored candidate lists using the saved, frozen embeddings."""
    embeddings = checkpoint["embeddings"]
    loan_id = checkpoint["loan_id"]
    partner_id = checkpoint["partner_id"]
    scored = {}
    for loan, pairs in grouped.items():
        loan_vector = embeddings[loan_id[loan]]
        values = []
        for partner, label in pairs:
            score = float(torch.dot(loan_vector, embeddings[partner_id[partner]]))
            values.append((partner, label, score))
        scored[loan] = values
    return scored


def ranking_metrics(scored):
    """Calculate AUC, AP, Hits@5 and NDCG at the requested cutoffs."""
    auc_values, ap_values, hits_values = [], [], []
    ndcg_values = {k: [] for k in (5, 10, 15, 20)}
    for values in scored.values():
        positives = [score for _, label, score in values if label]
        negatives = [score for _, label, score in values if not label]
        if not positives or not negatives:
            continue
        auc_values.append(sum(float(p > n) + .5 * float(p == n)
                              for p in positives for n in negatives) /
                         (len(positives) * len(negatives)))
        ranked = sorted(values, key=lambda item: item[2], reverse=True)
        seen_positive = 0
        precision_sum = 0.0
        for rank, (_, label, _) in enumerate(ranked, 1):
            if label:
                seen_positive += 1
                precision_sum += seen_positive / rank
        ap_values.append(precision_sum / len(positives))
        hits_values.append(float(any(label for _, label, _ in ranked[:5])))
        for k in ndcg_values:
            top = ranked[:k]
            dcg = sum(label / math.log2(index + 2) for index, (_, label, _) in enumerate(top))
            ideal = sum(1 / math.log2(index + 2) for index in range(min(k, len(positives))))
            ndcg_values[k].append(dcg / ideal if ideal else 0.0)
    average = lambda values: sum(values) / len(values) if values else 0.0
    result = {"loan_queries_evaluated": len(auc_values), "auc": average(auc_values),
              "average_precision": average(ap_values), "hits_at_5": average(hits_values)}
    for k, values in ndcg_values.items():
        result[f"ndcg_at_{k}"] = average(values)
    return result


def fairness_metrics(scored, test_rows, attribute, cutoff=5, threshold=.05):
    """Measure partner exposure gaps across incoming-loan groups.

    A group always receives ``cutoff`` recommendations, so comparing total
    selected slots would be trivially equal. Instead, for every partner we
    compare the probability that a loan from each group selects that partner.
    """
    row_lookup = {row["loan"]: row for row in test_rows}
    partner_names = {row["partner"]: row.get("partner_name", "unknown")
                     for row in test_rows}
    group_query_count = defaultdict(int)
    partner_group_selected = defaultdict(lambda: defaultdict(int))
    for loan, values in scored.items():
        # The query is a loan; its protected attribute labels this query group.
        group = row_value(row_lookup[loan], attribute)
        ranked = sorted(values, key=lambda item: item[2], reverse=True)
        group_query_count[group] += 1
        for partner, _, _ in ranked[:cutoff]:
            partner_group_selected[partner][group] += 1
    partner_gaps = []
    partner_weights = []
    partner_fairness = {}
    for partner, selected_by_group in partner_group_selected.items():
        rates = {group: selected_by_group.get(group, 0) / count
                 for group, count in group_query_count.items() if count}
        if len(rates) > 1:
            partner_gap = max(rates.values()) - min(rates.values())
            partner_gaps.append(partner_gap)
            # Weight partners by how often they actually appear in top-k lists.
            partner_weights.append(sum(selected_by_group.values()))
            partner_fairness[partner] = {
                "partner_id": partner,
                "partner_name": partner_names.get(partner, "unknown"),
                "group_selection_rates": rates,
                "group_selected_counts": dict(selected_by_group),
                "group_query_counts": dict(group_query_count),
                "selection_rate_gap": partner_gap,
                "threshold": threshold,
                "meets_threshold": partner_gap <= threshold,
            }
    # Do not expose an aggregate slot rate here: every loan receives exactly
    # cutoff recommendations, so that aggregate is identical by construction.
    average_gap = sum(partner_gaps) / len(partner_gaps) if partner_gaps else 0.0
    total_weight = sum(partner_weights)
    weighted_gap = (sum(gap * weight for gap, weight in zip(partner_gaps, partner_weights)) /
                    total_weight if total_weight else 0.0)
    # Use the weighted average as the primary benchmark rather than the
    # maximum single-partner gap, which is highly sensitive to rare partners.
    violations = sum(g > threshold for g in partner_gaps)
    return {"attribute": attribute, "cutoff": cutoff,
            "group_query_counts": dict(group_query_count),
            "selection_rate_gap": weighted_gap,
            "average_partner_gap": average_gap,
            "weighted_partner_gap": weighted_gap,
            "maximum_partner_gap": max(partner_gaps) if partner_gaps else 0.0,
            "partners_meeting_threshold": sum(item["meets_threshold"]
                                               for item in partner_fairness.values()),
            "partner_threshold_pass_rate": (
                sum(item["meets_threshold"] for item in partner_fairness.values()) /
                len(partner_fairness) if partner_fairness else 0.0),
            "per_partner": partner_fairness,
            "threshold": threshold,
            "loan_lists_evaluated": len(scored),
            "raw_violation_count": violations,
            "raw_violation_rate": violations / len(partner_gaps) if partner_gaps else 0.0,
            "partners_compared": len(partner_gaps),
            "guardrail_status": "measurement_only_not_enforced",
            "note": "For each partner, selection rates are compared across loan groups."}


def rerank_until_variance(scored_values, test_rows, attribute, minimum_k=10,
                          batch_size=10, variance_threshold=.0025):
    """Return at least ten partner recommendations for one loan query.

    Protected attributes belong to the query loan, so fairness is measured
    across many loan queries in ``fairness_metrics`` rather than within one
    list.  This function therefore preserves score order and enforces the
    minimum output size without pretending that a single loan has variance.
    """
    minimum_k = max(10, minimum_k)
    ranked = sorted(scored_values, key=lambda item: item[2], reverse=True)
    selected = ranked[:min(minimum_k, len(ranked))]
    return selected, {"stopped_at": len(selected), "variance": None,
                      "batches_checked": 1, "status": "score_order_only"}


def main():
    # Separate output paths keep verbose fairness details out of model results.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", type=Path, default=Path("outputs/models/gnn_model.pt"))
    parser.add_argument("--model-results", type=Path, default=Path("outputs/evaluation/model_evaluation.json"))
    parser.add_argument("--fairness-dir", type=Path, default=Path("outputs/evaluation/fairness"))
    parser.add_argument("--summary-file", type=Path, default=Path("outputs/evaluation/benchmark_summary.json"))
    parser.add_argument("--inference-results", type=Path,
                        default=Path("outputs/evaluation/reranked_inference.json"))
    parser.add_argument("--minimum-k", type=int, default=10,
                        help="Minimum recommendations returned for each partner")
    parser.add_argument("--eval-batch-size", type=int, default=10,
                        help="Additional candidates inspected per inference batch")
    parser.add_argument("--variance-threshold", type=float, default=.0025,
                        help="Maximum group selection-rate variance for early stopping")
    parser.add_argument("--seed", type=int, default=50413)
    args = parser.parse_args()

    # The checkpoint contains frozen embeddings and the exact temporal test split.
    try:
        # Newer PyTorch versions require an explicit opt-in for Python objects.
        checkpoint = torch.load(args.model_file, map_location="cpu", weights_only=False)
    except TypeError:
        # Older versions do not have the weights_only argument.
        checkpoint = torch.load(args.model_file, map_location="cpu")
    test_rows = checkpoint["test_rows"]
    # Inference direction: a new loan is the query and partners are ranked.
    grouped = candidates_by_loan(test_rows, args.seed)
    scored = score_candidates(checkpoint, grouped)

    model_result = ranking_metrics(scored)
    model_result.update({"test_loans": len(test_rows), "epochs_in_checkpoint": checkpoint.get("epochs")})
    args.model_results.parent.mkdir(parents=True, exist_ok=True)
    args.model_results.write_text(json.dumps(model_result, indent=2), encoding="utf-8")

    fairness_result = {}
    reranked_recommendations = {}
    partner_names = {row["partner"]: row.get("partner_name", "unknown")
                     for row in test_rows}
    args.fairness_dir.mkdir(parents=True, exist_ok=True)
    # Gender is the only fairness attribute used here. Region is omitted
    # because partners operate in specific geographies, making region gaps a
    # partner-eligibility effect rather than a clean fairness comparison.
    for attribute in ("gender",):
        result = fairness_metrics(scored, test_rows, attribute,
                                  cutoff=max(10, args.minimum_k))
        # Run inference-time reranking separately from the raw model metrics.
        rerank_stats = []
        recommendations = {}
        for partner, values in scored.items():
            selected, stats = rerank_until_variance(
                values, test_rows, attribute, args.minimum_k,
                args.eval_batch_size, args.variance_threshold)
            rerank_stats.append(stats)
            recommendations[partner] = [
                {"partner_id": item[0], "partner_name": partner_names.get(item[0], "unknown")}
                for item in selected]
        result["reranking"] = {
            "minimum_k": max(10, args.minimum_k),
            "eval_batch_size": args.eval_batch_size,
            "variance_threshold": args.variance_threshold,
            "average_stopped_at": (sum(item["stopped_at"] for item in rerank_stats) /
                                    len(rerank_stats) if rerank_stats else 0),
            "maximum_stopped_at": max((item["stopped_at"] for item in rerank_stats), default=0),
            "average_final_variance": None,
            "early_stop_lists": 0,
            "lists_evaluated": len(rerank_stats),
            "status": "reranking_disabled_for_query_level_attributes",
        }
        fairness_result[attribute] = result
        reranked_recommendations[attribute] = recommendations
        (args.fairness_dir / f"fairness_{attribute}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")

    # Save a compact inference audit instead of printing every recommendation.
    inference_output = {"settings": {"minimum_k": max(10, args.minimum_k),
                                      "eval_batch_size": args.eval_batch_size,
                                      "variance_threshold": args.variance_threshold},
                        "summary": {attribute: result["reranking"]
                                    for attribute, result in fairness_result.items()},
                        "recommendations": reranked_recommendations}
    args.inference_results.parent.mkdir(parents=True, exist_ok=True)
    args.inference_results.write_text(json.dumps(inference_output, indent=2), encoding="utf-8")

    # These flags make the proposal's targets explicit without hiding failures.
    summary = {"model": {"ndcg_at_5": model_result["ndcg_at_5"],
                          "target_ndcg_at_5": .80,
                          "ndcg_at_5_met": model_result["ndcg_at_5"] >= .80,
                          "auc_above_random": model_result["auc"] > .50},
               "fairness": {attribute: {
                   "selection_rate_gap": result["selection_rate_gap"],
                   "average_partner_gap": result["average_partner_gap"],
                   "weighted_partner_gap": result["weighted_partner_gap"],
                   "partners_meeting_threshold": result["partners_meeting_threshold"],
                   "partner_threshold_pass_rate": result["partner_threshold_pass_rate"],
                   "target_gap": .05,
                   "selection_gap_met": result["selection_rate_gap"] <= .05,
                   "raw_violation_rate": result["raw_violation_rate"],
                   "reranking": result["reranking"],
                   "guardrail_violation_rate": None,
                   "guardrail_status": result["guardrail_status"]}
                   for attribute, result in fairness_result.items()}}
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    args.summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Print only headline metrics; detailed rankings remain in JSON files.
    print(f"loan queries evaluated: {model_result['loan_queries_evaluated']:,}")
    print(f"AUC: {model_result['auc']:.4f} | AP: {model_result['average_precision']:.4f} | "
          f"Hits@5: {model_result['hits_at_5']:.4f}")
    print("NDCG@5={:.4f} | NDCG@10={:.4f} | NDCG@15={:.4f} | NDCG@20={:.4f}".format(
        model_result["ndcg_at_5"], model_result["ndcg_at_10"],
        model_result["ndcg_at_15"], model_result["ndcg_at_20"]))
    for attribute, result in fairness_result.items():
        print(f"{attribute} selection-rate gap@{result['cutoff']}: "
              f"weighted={result['selection_rate_gap']:.4f} | "
              f"partners meeting threshold="
              f"{result['partners_meeting_threshold']}/{result['partners_compared']}")
    print(json.dumps({"model_results": str(args.model_results),
                      "fairness_results": str(args.fairness_dir),
                      "benchmark_summary": str(args.summary_file)}, indent=2))


if __name__ == "__main__":
    main()
