"""Evaluation logic for Hit@1, Evidence Hit@3, and benchmark reporting."""
import re
from typing import Any


def check_hit(question: dict[str, Any], chunk: dict[str, Any]) -> bool:
    """Verify if chunk matches gold source/page and contains sufficient gold evidence."""
    gold_page = question["gold_page"]
    gold_evidence = question["evidence_quote"]
    chunk_meta = chunk["metadata"]
    
    # 1. Page verification
    if chunk_meta.get("page") != gold_page:
        return False
        
    # 2. Evidence verification: clean whitespace/hyphenation
    ev_norm = re.sub(r'\s+', ' ', gold_evidence).lower().strip()
    chunk_norm = re.sub(r'\s+', ' ', chunk["text"]).lower().strip()
    
    # Check if a substantial part (or entire) evidence quote is contained
    # Taking key phrases or first 45 chars
    key_prefix = ev_norm[:45].strip()
    if key_prefix in chunk_norm:
        return True
        
    # Check word overlap
    ev_words = set(re.findall(r'\b\w{4,}\b', ev_norm))
    chunk_words = set(re.findall(r'\b\w{4,}\b', chunk_norm))
    if len(ev_words) > 0 and len(ev_words.intersection(chunk_words)) / len(ev_words) >= 0.75:
        return True
        
    return False


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Check Hit@1 and Evidence Hit@3 for a single question."""
    hit_at_1 = False
    hit_at_3 = False
    hit_positions = []
    
    for rank, hit in enumerate(hits, start=1):
        is_match = check_hit(question, hit)
        hit["is_evidence_hit"] = is_match
        if is_match:
            hit_positions.append(rank)
            if rank == 1:
                hit_at_1 = True
            if rank <= 3:
                hit_at_3 = True

    return {
        "question_id": question["id"],
        "split": question["split"],
        "question": question["question"],
        "gold_page": question["gold_page"],
        "gold_evidence": question["evidence_quote"],
        "filter_applied": question.get("filter"),
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "first_hit_rank": hit_positions[0] if hit_positions else None,
        "hits": hits
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 across splits."""
    dev_traces = [t for t in traces if t["split"] == "dev"]
    test_traces = [t for t in traces if t["split"] == "test"]
    
    def calc_stats(sub_traces: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(sub_traces)
        if n == 0:
            return {"count": 0, "hit1": "0/0 (0.0%)", "hit3": "0/0 (0.0%)"}
        h1 = sum(1 for t in sub_traces if t["hit_at_1"])
        h3 = sum(1 for t in sub_traces if t["hit_at_3"])
        return {
            "count": n,
            "hit1_num": h1,
            "hit1_den": n,
            "hit1_str": f"{h1}/{n} ({h1/n*100:.1f}%)",
            "hit3_num": h3,
            "hit3_den": n,
            "hit3_str": f"{h3}/{n} ({h3/n*100:.1f}%)"
        }
        
    return {
        "overall": calc_stats(traces),
        "dev": calc_stats(dev_traces),
        "test": calc_stats(test_traces)
    }
