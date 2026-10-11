"""Model requests, retries, exact response validation and scoring primitives."""

import json
import os
import re
import time

from data import TYPES, tokenize, unique_entities, locate_entity

RESPONSE_PARSER_VERSION = "outer_json_fence_v1"
_FENCE = re.compile(r"\A[ \t\r\n]*```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```[ \t\r\n]*\Z", re.IGNORECASE)


def unwrap_json_fence(raw):
    """Remove one whole-response fence only; retain its body verbatim."""
    if not isinstance(raw, str):
        return raw
    match = _FENCE.fullmatch(raw)
    if match and "```" not in match["body"]:
        return match["body"]
    return raw


def entity_list(data, sentence, tokens, types):
    if not isinstance(data, dict) or not isinstance(data.get("entities"), list):
        raise ValueError('Return an object with an "entities" list.')
    if any(not isinstance(item, dict) for item in data["entities"]):
        raise ValueError("Every entity must be an object.")
    return unique_entities([locate_entity(item, sentence, tokens, types)
                            for item in data["entities"]])


def decisions(data, count):
    items = data.get("decisions") if isinstance(data, dict) else None
    if not isinstance(items, list) or any(not isinstance(x, dict) for x in items):
        raise ValueError('Return an object with a "decisions" list.')
    ids = [x.get("id") for x in items]
    if any(type(i) is not int for i in ids) or sorted(ids) != list(range(count)):
        raise ValueError("Return exactly one decision for every supplied candidate ID.")
    if any(not isinstance(x.get("reason"), str) or not x["reason"].strip() for x in items):
        raise ValueError("Each decision needs one short reason.")
    return sorted(items, key=lambda x: x["id"])


def check_evidence(item, payload, config):
    """Validate evidence references for edits; this does not prove semantic correctness."""
    quote = item.get("evidence_text")
    ids = item.get("example_ids", [])
    allowed = {e["id"] for e in payload.get("training_examples", [])}
    rule = item.get("rule_id")
    if not isinstance(quote, str) or not quote.strip() or quote not in payload["sentence"]:
        raise ValueError("An edit needs evidence_text quoted exactly from the sentence.")
    if not isinstance(ids, list) or any(not isinstance(i, str) or i not in allowed for i in ids):
        raise ValueError("example_ids must refer to supplied training examples.")
    if rule is not None and (type(rule) is not int or not 0 <= rule < config["_rule_count"]):
        raise ValueError("rule_id must refer to a supplied task rule.")
    if not ids and rule is None:
        raise ValueError("An edit needs at least one example_id or rule_id.")


def evidence(item, payload):
    quote = item.get("evidence_text")
    if not isinstance(quote, str) or not quote.strip() or quote not in payload["sentence"]:
        raise ValueError("evidence_text must be a nonempty exact quote from the sentence.")
    if not isinstance(item.get("reason"), str) or not item["reason"].strip():
        raise ValueError("Provide a short specific reason.")


def make_client(config):
    from openai import OpenAI
    key = config.get("api_keys") or os.getenv(config.get("api_key_env", ""))
    if not key:
        raise ValueError("Set api_keys in the config or the configured API key environment variable.")
    return OpenAI(api_key=key, base_url=config["base_url"],
                  timeout=config.get("timeout", 120), max_retries=0)


def call_agent(stage, payload, config, client, validate):
    """Retry until validated success when max_attempts is None; Ctrl+C can cancel."""
    from openai import APIConnectionError, APIStatusError

    messages = [{"role": "system", "content": config["_prompts"][stage]},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    trace = {"stage": stage, "status": "failed", "attempts": [],
             "training_examples": payload.get("training_examples", [])}
    limit = config.get("max_attempts", 3)
    attempt = 0
    while limit is None or attempt < limit:
        time.sleep(config.get("sleep_seconds", 0))
        started = time.monotonic()
        entry = {"attempt": attempt + 1, "raw_response": None, "error": None,
                 "response_parser_version": RESPONSE_PARSER_VERSION}
        retryable = True
        retry_after = min(config.get("retry_delay", 2) * 2 ** min(attempt, 10), 30)
        value = None
        try:
            extra = {}
            if "enable_thinking" in config:
                extra["extra_body"] = {"enable_thinking": bool(config["enable_thinking"])}
            response = client.chat.completions.create(
                model=config.get("api_model_name", config["model_name"]), messages=messages,
                temperature=config.get("temperature", 1), **extra)
            entry["raw_response"] = response.choices[0].message.content
            entry["usage"] = response.usage.model_dump() if response.usage else None
            if response.choices[0].finish_reason == "length":
                raise ValueError("Response was truncated; return a shorter complete JSON object.")
            raw = entry["raw_response"]
            if not isinstance(raw, str):
                raise ValueError("Expected JSON text in the response.")
            normalized = unwrap_json_fence(raw)
            entry["outer_json_fence_removed"] = normalized != raw
            value = validate(json.loads(normalized))
            trace["status"] = "ok"
        except APIStatusError as exc:
            entry["error"] = f"HTTP {exc.status_code}"
            secret = config.get("api_keys") or os.getenv(config.get("api_key_env", ""))
            detail = exc.response.text
            entry["error_detail"] = (detail.replace(secret, "[REDACTED]") if secret else detail)[:2000]
            trace["fatal"] = limit is not None and exc.status_code in (400, 401, 403, 404)
            retryable = limit is None or exc.status_code in (408, 409, 429) or exc.status_code >= 500
            try:
                retry_after = max(retry_after, min(float(exc.response.headers.get("retry-after", 0)), 120))
            except ValueError:
                pass
        except APIConnectionError:
            entry["error"] = "APIConnectionError/timeout"
        except (ValueError, TypeError, KeyError, IndexError, StopIteration) as exc:
            entry["error"] = "Invalid response: " + str(exc)[:300]
            previous = entry["raw_response"]
            messages = messages[:2] + ([{"role": "assistant", "content": previous[:16000]}]
                                       if isinstance(previous, str) and previous else []) + [{"role": "user", "content":
                "The previous response was invalid: " + entry["error"] +
                " Return complete corrected JSON; preserve valid entities and fix the reported defect."}]
        entry["elapsed_seconds"] = round(time.monotonic() - started, 3)
        trace["attempts"].append(entry)
        if config.get("_attempt_logger"):
            config["_attempt_logger"].info("stage=%s attempt=%s", stage, json.dumps(entry, ensure_ascii=False))
        if trace["status"] == "ok":
            return value, trace
        if not retryable:
            break
        attempt += 1
        if limit is None or attempt < limit:
            time.sleep(retry_after)
    return None, trace


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
