"""
Packing stage: concatenate tokenized docs into fixed-length sequences.

Two gates run before packing, both firewalled and both counted, not silent:

  1. Evaluation/validation firewall -- split=eval docs are rejected here,
     never packed for training.
  2. Quality gate -- quality_reject_final (computed in shards.py, stored
     ON the shard record) rejects low-scoring docs UNLESS the doc's lane
     is protected (see quality.py), in which case it's kept and the
     override is counted separately.

Packing policy differs by lane, per the session's explicit ask ("packing
policy for different types of data... for coding it's going to be
different"):

  - code: longer sequence length (functions run longer than a sentence),
    and NEVER truncated -- an oversized function is skipped rather than
    cut mid-function, since a truncated function is worse than no
    function (it would teach broken syntax).
  - en / hi (prose lanes): shorter sequence length, truncate-on-overflow
    is acceptable since a cut sentence is still readable training signal.

Packing runs per-lane (each lane gets its own pack pool with its own
seq_len), so a single pack never mixes code with prose.
"""
import json
from pathlib import Path

LANE_PACKING_POLICY = {
    "code": {"seq_len": 256, "on_oversized": "skip"},
    "en":   {"seq_len": 128, "on_oversized": "truncate"},
    "hi":   {"seq_len": 128, "on_oversized": "truncate"},
}
DEFAULT_POLICY = {"seq_len": 128, "on_oversized": "truncate"}


def load_train_docs(manifest, pad_id, tokenizer):
    """
    Read shard records and apply the full OPUS + eval gate pipeline:
      - eval firewall (split == 'eval' -> rejected, never trained on)
      - OPUS decision (accept / reject / defer), with protected-lane
        override -- a protected doc that would be rejected OR deferred
        is accepted anyway, and the override is counted separately.
    Deferred docs (uncertain, not protected) are held OUT of this run --
    not trained on, not discarded either; a real system would route them
    to further review rather than making an immediate accept/reject call.
    Returns (accepted, eval_rejected, quality_rejected, deferred, protected_overrides)
    """
    accepted, eval_rejected, quality_rejected, deferred, protected_overrides = [], [], [], [], []
    for entry in manifest:
        for line in Path(entry["path"]).read_text().splitlines():
            rec = json.loads(line)
            if rec["split"] == "eval":
                eval_rejected.append(rec)
                continue
            decision = rec.get("opus_decision_final", "accept" if not rec.get("quality_reject_final") else "reject")
            was_overridden = rec.get("opus_decision") != "accept" and decision == "accept"
            if was_overridden:
                protected_overrides.append(rec)  # kept via protected-lane override, tracked
            if decision == "reject":
                quality_rejected.append(rec)
                continue
            if decision == "defer":
                deferred.append(rec)
                continue
            accepted.append(rec)
    return accepted, eval_rejected, quality_rejected, deferred, protected_overrides


def _pack_one_lane(docs, pad_id, seq_len, on_oversized):
    """Greedy packing within a single lane's sequence-length policy."""
    packed = []
    buf_ids, buf_loss, buf_pos, buf_docs = [], [], [], []
    skipped_oversized = 0

    def flush():
        nonlocal buf_ids, buf_loss, buf_pos, buf_docs
        if not buf_ids:
            return
        pad_needed = seq_len - len(buf_ids)
        ids = buf_ids + [pad_id] * pad_needed
        loss_mask = buf_loss + [0] * pad_needed
        attn_mask = [1] * len(buf_ids) + [0] * pad_needed
        pos_ids = buf_pos + [0] * pad_needed
        packed.append({
            "input_ids": ids,
            "loss_mask": loss_mask,
            "attention_mask": attn_mask,
            "position_ids": pos_ids,
            "doc_ids": list(buf_docs),
            "seq_len": seq_len,
        })
        buf_ids, buf_loss, buf_pos, buf_docs = [], [], [], []

    for rec in docs:
        ids = rec["token_ids"]
        if len(ids) > seq_len:
            if on_oversized == "skip":
                skipped_oversized += 1
                continue
            ids = ids[:seq_len]  # truncate
        if len(buf_ids) + len(ids) > seq_len:
            flush()
        buf_ids.extend(ids)
        buf_loss.extend([1] * len(ids))
        buf_pos.extend(range(len(ids)))
        buf_docs.append(rec["doc_id"])
    flush()
    return packed, skipped_oversized


def pack_sequences(docs, pad_id, lane_policy=None):
    """
    Splits docs by lane, packs each lane with its own policy, returns
    {lane: {"packs": [...], "skipped_oversized": n}} plus a flat list.
    """
    policy = lane_policy or LANE_PACKING_POLICY
    by_lane = {}
    for rec in docs:
        by_lane.setdefault(rec["lane"], []).append(rec)

    per_lane_result = {}
    all_packed = []
    for lane, lane_docs in by_lane.items():
        p = policy.get(lane, DEFAULT_POLICY)
        packed, skipped = _pack_one_lane(lane_docs, pad_id, p["seq_len"], p["on_oversized"])
        for pk in packed:
            pk["lane"] = lane
        per_lane_result[lane] = {
            "packs": packed,
            "skipped_oversized": skipped,
            "policy": p,
        }
        all_packed.extend(packed)
    return all_packed, per_lane_result
