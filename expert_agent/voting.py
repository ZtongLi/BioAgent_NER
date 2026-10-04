"""Deterministic counting and explicit task-level overlap policy."""


def conflict_pairs(candidates, overlap_policy):
    if overlap_policy not in ("forbid", "allow_nested"):
        raise ValueError("overlap_policy must be forbid or allow_nested")
    pairs = []
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            overlap = max(a["start"], b["start"]) < min(a["end"], b["end"])
            same_span = (a["start"], a["end"]) == (b["start"], b["end"])
            nested = ((a["start"] <= b["start"] and b["end"] <= a["end"])
                      or (b["start"] <= a["start"] and a["end"] <= b["end"]))
            if overlap and (overlap_policy == "forbid" or same_span or not nested):
                pairs.append([a["id"], b["id"]])
    return pairs


def tally(candidates, ballots):
    if len(ballots) != 3:
        raise ValueError("Exactly three successful expert ballots are required.")
    expected = list(range(len(candidates)))
    for items in ballots.values():
        if sorted(x["id"] for x in items) != expected:
            raise ValueError("Each expert must vote exactly once on every candidate.")
    counts = []
    for candidate in candidates:
        votes = [next(x["vote"] for x in items if x["id"] == candidate["id"])
                 for items in ballots.values()]
        if any(v not in ("support", "oppose", "abstain") for v in votes):
            raise ValueError("Invalid vote")
        count = {v: votes.count(v) for v in ("support", "oppose", "abstain")}
        status = "accepted" if count["support"] >= 2 else "rejected" if count["oppose"] >= 2 else "unresolved"
        counts.append({"id": candidate["id"], **count, "status": status})
    return counts


def constraints(counts, pairs):
    """A majority-supported candidate stays locked unless another non-rejected one conflicts."""
    status = {x["id"]: x["status"] for x in counts}
    contested = {i for pair in pairs if all(status[i] != "rejected" for i in pair) for i in pair}
    return {i: "drop" if value == "rejected" else "keep"
            for i, value in status.items() if value == "rejected" or (value == "accepted" and i not in contested)}


def vote_selection(counts, pairs):
    accepted = {x["id"] for x in counts if x["status"] == "accepted"}
    conflicting = {i for pair in pairs if all(i in accepted for i in pair) for i in pair}
    return accepted - conflicting
