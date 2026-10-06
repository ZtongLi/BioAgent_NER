"""Shared entity-level scoring and offline expert diagnostics; no inference dependencies."""

def precision_recall_f1(tp, fp, fn, *, empty_score=0.0, harmonic_f1=False):
    """Entity micro PRF. Preserve each caller's historical floating-point arithmetic."""
    precision = tp / (tp + fp) if tp + fp else empty_score
    recall = tp / (tp + fn) if tp + fn else empty_score
    if harmonic_f1:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else empty_score
    else:
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else empty_score
    return precision, recall, f1


def calculate_metrics(true_positive, predicted_total, gold_total):
    """Baseline-compatible API; valid empty predictions still participate in scoring."""
    return precision_recall_f1(true_positive, predicted_total - true_positive,
                              gold_total - true_positive, empty_score=0, harmonic_f1=True)


def span_set(entities):
    return {(*e["pos"], e["type"]) if "pos" in e else (e["start"], e["end"], e["type"])
            for e in entities or []}


def successful_record(row):
    stages = row.get("stages", [])
    return row.get("fully_executed") is True and bool(stages) and all(
        s["status"] in ("ok", "skipped_empty") for s in stages)


def evaluate_ner(records):
    """Score fully successful samples; failed requests/parsing are neither FP nor FN."""
    tp = fp = fn = evaluated = 0
    effects = {}
    for row in records:
        stages = row.get("stages", [])
        success = successful_record(row)
        if success:
            evaluated += 1
            gold, pred = span_set(row["gold_entities"]), span_set(row["pred_entities"])
            tp += len(gold & pred)
            fp += len(pred - gold)
            fn += len(gold - pred)
        for stage in stages:
            totals = effects.setdefault(stage["stage"], dict.fromkeys(
                ["samples", "failed_samples", "evaluated_samples", "true_added", "false_added",
                 "true_removed", "false_removed"], 0))
            totals["samples"] += 1
            totals["failed_samples"] += int(stage["status"] == "failed")
            if not success:
                continue
            totals["evaluated_samples"] += 1
            before, after = span_set(stage["input_entities"]), span_set(stage["output_entities"])
            added, removed = after - before, before - after
            for key, value in {"true_added": len(added & gold), "false_added": len(added - gold),
                               "true_removed": len(removed & gold), "false_removed": len(removed - gold)}.items():
                totals[key] += value
    empty_score = 0.0 if evaluated else None
    precision, recall, f1 = precision_recall_f1(tp, fp, fn, empty_score=empty_score)
    return {"precision": precision, "recall": recall, "f1": f1,
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "num_samples": len(records),
            "failed_samples": len(records) - evaluated,
            "evaluated_samples": evaluated,
            "success_rate": evaluated / len(records) if records else None,
            "evaluation_scope": "successful_samples_only",
            "stage_effects": effects}



def expert_diagnostics(records):
    """Gold-dependent diagnostics live only in the offline evaluator."""
    totals = dict.fromkeys(("unique_expert_true_candidates", "unique_expert_true_retained",
                           "vote_true_rejected", "vote_false_accepted",
                           "aggregation_true_added", "aggregation_false_added",
                           "aggregation_true_removed", "aggregation_false_removed"), 0)
    for row in records:
        if not successful_record(row):
            continue
        gold, final = span_set(row["gold_entities"]), span_set(row["pred_entities"])
        candidates = {c["id"]: (c["start"], c["end"], c["type"]) for c in row.get("candidates", [])}
        for c in row.get("candidates", []):
            if len(c.get("sources", [])) == 1 and candidates[c["id"]] in gold:
                totals["unique_expert_true_candidates"] += 1
                totals["unique_expert_true_retained"] += int(candidates[c["id"]] in final)
        for count in row.get("vote_counts", []):
            truth = candidates[count["id"]] in gold
            totals["vote_true_rejected"] += int(truth and count["status"] == "rejected")
            totals["vote_false_accepted"] += int(not truth and count["status"] == "accepted")
        for stage in row["stages"]:
            if stage["stage"] != "aggregation":
                continue
            before, after = span_set(stage["input_entities"]), span_set(stage["output_entities"])
            for action, changed in (("added", after - before), ("removed", before - after)):
                totals[f"aggregation_true_{action}"] += len(changed & gold)
                totals[f"aggregation_false_{action}"] += len(changed - gold)
    count = totals["unique_expert_true_candidates"]
    totals["unique_expert_true_retention_rate"] = totals["unique_expert_true_retained"] / count if count else None
    return totals

