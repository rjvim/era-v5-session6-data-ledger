"""
Turns raw docs into IMMUTABLE tokenized shards + a manifest.

Immutable = once written, a shard's bytes and hash never change.
If you rerun this on the same input with the same frozen tokenizer,
you get byte-identical shards (same hash) -> reproducibility.
"""
import json
import hashlib
from pathlib import Path

from quality import evaluate as quality_evaluate


def content_hash(token_ids):
    raw = json.dumps(token_ids).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def write_shards(docs, tokenizer, shard_dir, docs_per_shard=20):
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for shard_idx in range(0, len(docs), docs_per_shard):
        chunk = docs[shard_idx: shard_idx + docs_per_shard]
        records = []
        for d in chunk:
            ids = tokenizer.encode(d["text"])
            q = quality_evaluate(d)  # deterministic function of text+lane -> reproducible
            records.append({
                "doc_id": d["doc_id"],
                "lane": d["lane"],
                "split": d["split"],
                "token_ids": ids,
                "num_tokens": len(ids),
                # quality / protected-flow tags, stored ON the shard record
                # itself per the session's "does your shard store the tag"
                # requirement -- not computed later and bolted on.
                "quality_score": q["quality_score"],
                "opus_decision": q["opus_decision"],
                "protected": q["protected"],
                "opus_decision_final": q["opus_decision_final"],
                # legacy fields, kept for packing.py compatibility
                "quality_reject": q["quality_reject"],
                "quality_reject_final": q["quality_reject_final"],
            })
        # hash is over the fully-resolved shard content -> immutability check
        shard_hash = content_hash([r["token_ids"] for r in records])
        shard_id = f"shard_{shard_idx // docs_per_shard:04d}_{shard_hash}"
        shard_path = shard_dir / f"{shard_id}.jsonl"

        with open(shard_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        lanes_in_shard = sorted(set(r["lane"] for r in records))
        splits_in_shard = sorted(set(r["split"] for r in records))
        n_quality_reject_final = sum(1 for r in records if r["quality_reject_final"])
        n_protected_overrides = sum(1 for r in records if r["quality_reject"] and r["protected"])
        manifest.append({
            "shard_id": shard_id,
            "path": str(shard_path),
            "hash": shard_hash,
            "tokenizer_vocab_hash": tokenizer.vocab_hash,
            "num_docs": len(records),
            "num_tokens": sum(r["num_tokens"] for r in records),
            "lanes": lanes_in_shard,
            "splits": splits_in_shard,
            # eval firewall: any shard containing eval docs is flagged.
            # training loop must refuse to pull tokens from these.
            "eval_only": splits_in_shard == ["eval"],
            "contains_eval": "eval" in splits_in_shard,
            # quality / protected-flow summary for this shard
            "quality_rejects_final": n_quality_reject_final,
            "protected_overrides": n_protected_overrides,
        })

    manifest_path = shard_dir.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest, str(manifest_path)
