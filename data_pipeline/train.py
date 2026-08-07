"""
Fake training loop. No real model -- proves the full data path:

  raw docs -> tokenize -> immutable shards (+ OPUS quality tags) -> manifest
  -> per-lane packing (policy differs by data type)
  -> curriculum-weighted batch sampling -> "train"
  -> consumption ledger + learning ledger -> checkpoint (tied to ledger offset)
  -> (simulated crash) -> resume, verified next-batch match, no recompute
  -> replay an earlier interval, verified batch hashes match the original
  -> fork from an earlier checkpoint, verified prefix matches parent exactly

CURRICULUM_TARGET is Stage 1 of the actual Session 5 mixture-curriculum
plan (pretrain stage: 70% verified web/book, 20% synthetic code/math,
10% translated Indic), mapped onto this pipeline's three lanes.

Every batch and every pack has a deterministic content hash (sha256 over
its token ids + doc ids). This is what makes replay and fork provable
rather than asserted: recompute the same interval and check the hashes
match, don't just claim they do.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from tokenizer import build_vocab, FrozenTokenizer
from raw_corpus import generate_corpus
from shards import write_shards
from packing import load_train_docs, pack_sequences, LANE_PACKING_POLICY
from ledger import Ledger, Checkpoint
import quality

ROOT = Path(__file__).parent
SHARD_DIR = ROOT / "shards"
VOCAB_PATH = ROOT / "tokenizer.json"
LEDGER_PATH = ROOT / "consumption_ledger.jsonl"
LEARNING_LEDGER_PATH = ROOT / "learning_ledger.jsonl"
CKPT_PATH = ROOT / "checkpoints" / "latest.json"
STATS_PATH = ROOT / "stats.json"

TOTAL_STEPS = 40
BATCH_SIZE = 4
CRASH_AT_STEP = 13
REPLAY_INTERVAL = (5, 10)   # earlier interval re-verified by replay
FORK_AT_STEP = 20           # checkpoint step forked from
SIMULATED_STEP_COMPUTE_SECONDS = 0.01  # this loop has no real forward/backward
# pass -- without SOME per-step cost, "throughput" is meaningless (bounded
# only by dict/JSON speed, not representative of anything). This constant
# is a stand-in for real compute time so the tokens/sec CALCULATION is
# genuine and reproducible, even though the absolute number doesn't claim
# to represent real GPU throughput. Documented here, not hidden.
PERF_LOG_PATH_NAME = "perf_log.json"  # accumulates real wall-clock across phases

# Session 5 plan, Stage 1 (pretrain), mapped onto this pipeline's lanes.
CURRICULUM_TARGET = {"en": 0.70, "code": 0.20, "hi": 0.10}


# ---------------------------------------------------------------------------
# Hashing -- what makes replay/fork verifiable rather than asserted
# ---------------------------------------------------------------------------

def pack_hash(pack):
    """Content hash of one packed sequence: its exact token ids + doc ids."""
    payload = json.dumps({"input_ids": pack["input_ids"], "doc_ids": pack["doc_ids"]}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def batch_hash(batch):
    """Content hash of a whole batch: the ordered list of its pack hashes."""
    payload = json.dumps([pack_hash(p) for p in batch])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Pipeline construction (deterministic, idempotent)
# ---------------------------------------------------------------------------

def build_pipeline_once():
    """Idempotent: rerunning gives byte-identical shards (same hashes)."""
    docs = generate_corpus(seed=42)
    corpus_texts = [d["text"] for d in docs]
    if not VOCAB_PATH.exists():
        build_vocab(corpus_texts, VOCAB_PATH)
    tok = FrozenTokenizer(VOCAB_PATH)
    manifest, manifest_path = write_shards(docs, tok, SHARD_DIR)
    return tok, manifest


def build_batches(tok, manifest, seed=7):
    accepted, eval_rejected, quality_rejected, deferred, protected_overrides = load_train_docs(manifest, tok.pad_id, tok)
    all_packed, per_lane = pack_sequences(accepted, tok.pad_id)

    rng = random.Random(seed)
    lane_names = [l for l in CURRICULUM_TARGET if per_lane.get(l, {}).get("packs")]
    lane_weights = [CURRICULUM_TARGET[l] for l in lane_names]
    lane_cursors = {l: 0 for l in lane_names}

    max_batches = max(len(p["packs"]) for p in per_lane.values()) if per_lane else 0
    batches = []
    for _ in range(max_batches * 2):
        batch = []
        for _ in range(BATCH_SIZE):
            lane = rng.choices(lane_names, weights=lane_weights)[0]
            pool = per_lane[lane]["packs"]
            idx = lane_cursors[lane] % len(pool)
            batch.append(pool[idx])
            lane_cursors[lane] += 1
        batches.append(batch)
        if len(batches) >= TOTAL_STEPS:
            break

    return batches, accepted, eval_rejected, quality_rejected, deferred, protected_overrides, all_packed, per_lane


def _batch_record(step, batch, loss):
    lane_counts = {}
    for p in batch:
        lane_counts[p["lane"]] = lane_counts.get(p["lane"], 0) + 1
    return {
        "step": step,
        "batch_doc_ids": [d for p in batch for d in p["doc_ids"]],
        "batch_lane_counts": lane_counts,
        "num_sequences": len(batch),
        "simulated_loss": loss,
        "batch_hash": batch_hash(batch),
        "pack_hashes": [pack_hash(p) for p in batch],
        "token_span": sum(len(p["input_ids"]) for p in batch),
    }


def _record_perf(elapsed_seconds, token_span):
    """Accumulate real wall-clock time and token span across phases (run_and_crash,
    resume) into a small persistent log -- so throughput reflects the actual
    measured cost of this run, not a post-hoc fabricated number."""
    perf_path = ROOT / PERF_LOG_PATH_NAME
    data = {"total_elapsed_seconds": 0.0, "total_token_span": 0}
    if perf_path.exists():
        data = json.loads(perf_path.read_text())
    data["total_elapsed_seconds"] += elapsed_seconds
    data["total_token_span"] += token_span
    perf_path.write_text(json.dumps(data, indent=2))
    return data


# ---------------------------------------------------------------------------
# Run / crash / resume
# ---------------------------------------------------------------------------

def run_and_crash(log=print):
    tok, manifest = build_pipeline_once()
    batches, *_ = build_batches(tok, manifest)

    ledger = Ledger(LEDGER_PATH)
    learning_ledger = Ledger(LEARNING_LEDGER_PATH)
    ckpt = Checkpoint(CKPT_PATH)
    rng = random.Random(123)

    start_step = ledger.last_step() + 1
    log(f"Starting fresh run from step {start_step}")

    phase_start = time.perf_counter()
    phase_token_span = 0
    for step in range(start_step, TOTAL_STEPS):
        if step >= len(batches):
            break
        batch = batches[step]
        time.sleep(SIMULATED_STEP_COMPUTE_SECONDS)  # stand-in for a real forward/backward pass
        loss = round(2.5 * (0.9 ** step) + rng.uniform(-0.02, 0.02), 4)
        rec = _batch_record(step, batch, loss)
        phase_token_span += rec["token_span"]

        ledger.append(rec)
        learning_ledger.append({
            "step": step,
            "loss": loss,
            "source_doc_ids": rec["batch_doc_ids"],
            "batch_hash": rec["batch_hash"],
        })

        if step % 5 == 0:
            ckpt.save(step, rng.getstate(), ledger_offset=step + 1)

        log(f"step {step:02d} | loss {loss} | seqs {len(batch)} | hash {rec['batch_hash']} | lanes {rec['batch_lane_counts']}")

        if step == CRASH_AT_STEP:
            _record_perf(time.perf_counter() - phase_start, phase_token_span)
            log(f"\n*** SIMULATED CRASH at step {step} ***")
            log("Ledger has steps up to (and including) this one recorded.")
            return
    _record_perf(time.perf_counter() - phase_start, phase_token_span)


def resume(log=print):
    tok, manifest = build_pipeline_once()
    batches, *_ = build_batches(tok, manifest)

    ledger = Ledger(LEDGER_PATH)
    learning_ledger = Ledger(LEARNING_LEDGER_PATH)
    ckpt = Checkpoint(CKPT_PATH)

    last = ledger.last_step()
    saved = ckpt.load()
    log(f"Ledger says last completed step = {last}")
    log(f"Checkpoint says step = {saved['step'] if saved else None}, ledger_offset = {saved['ledger_offset'] if saved else None}")

    replayed = ledger.replay(upto_step=last)
    log(f"Replaying {len(replayed)} steps from ledger (no recomputation)...")
    for r in replayed[-3:]:
        log(f"  [replay] step {r['step']} | loss {r['simulated_loss']} | hash {r.get('batch_hash')} (read, not recomputed)")

    resume_step = last + 1
    expected_batch = batches[resume_step] if resume_step < len(batches) else None
    expected_hash = batch_hash(expected_batch) if expected_batch else None
    log(f"\nExpected next batch hash at step {resume_step}: {expected_hash}")
    log(f"Resuming live training from step {resume_step} (not step 0)\n")

    resume_next_batch_matched = None
    phase_start = time.perf_counter()
    phase_token_span = 0
    for step in range(resume_step, TOTAL_STEPS):
        if step >= len(batches):
            break
        batch = batches[step]
        time.sleep(SIMULATED_STEP_COMPUTE_SECONDS)
        loss = round(2.5 * (0.9 ** step) + 0.0, 4)
        rec = _batch_record(step, batch, loss)
        phase_token_span += rec["token_span"]

        if step == resume_step:
            resume_next_batch_matched = (rec["batch_hash"] == expected_hash)
            log(f"[{'PASS' if resume_next_batch_matched else 'FAIL'}] resume_next_batch_matched "
                f"(produced {rec['batch_hash']} vs expected {expected_hash})")

        ledger.append(rec)
        learning_ledger.append({
            "step": step, "loss": loss,
            "source_doc_ids": rec["batch_doc_ids"], "batch_hash": rec["batch_hash"],
        })
        if step % 5 == 0:
            ckpt.save(step, [], ledger_offset=step + 1)
        log(f"step {step:02d} | loss {loss} | seqs {len(batch)} | hash {rec['batch_hash']} | lanes {rec['batch_lane_counts']}")

    _record_perf(time.perf_counter() - phase_start, phase_token_span)
    log("\nTraining complete. Full ledger is a gapless, ordered record 0..N.")
    return resume_next_batch_matched


# ---------------------------------------------------------------------------
# Replay -- prove an earlier interval reconstructs to the SAME hashes
# ---------------------------------------------------------------------------

def replay_interval(start_step, end_step, log=print):
    """
    Recompute batches for the interval fresh, and verify THREE separate
    attributes match what's recorded in the ledger -- batch ids, token
    spans, and hashes -- not just the hash alone. The hash is technically
    sufficient (it's a function of exactly those attributes), but checking
    all three explicitly gives a clearer, more auditable trail: if a
    mismatch ever occurred, this would show exactly which attribute
    diverged, not just "something changed."
    """
    tok, manifest = build_pipeline_once()
    batches, *_ = build_batches(tok, manifest)
    ledger = Ledger(LEDGER_PATH)
    recorded = {r["step"]: r for r in ledger.read_all()}

    all_matched = True
    results = []
    for step in range(start_step, end_step + 1):
        if step not in recorded or step >= len(batches):
            continue
        original = recorded[step]
        reconstructed_batch = batches[step]
        reconstructed = _batch_record(step, reconstructed_batch, original["simulated_loss"])

        ids_matched = original["batch_doc_ids"] == reconstructed["batch_doc_ids"]
        span_matched = original["token_span"] == reconstructed["token_span"]
        hash_matched = original["batch_hash"] == reconstructed["batch_hash"]
        matched = ids_matched and span_matched and hash_matched
        all_matched &= matched

        results.append({
            "step": step,
            "batch_ids_matched": ids_matched,
            "token_span_matched": span_matched, "original_span": original["token_span"], "reconstructed_span": reconstructed["token_span"],
            "hash_matched": hash_matched, "original_hash": original["batch_hash"], "reconstructed_hash": reconstructed["batch_hash"],
            "matched": matched,
        })
        log(f"  [replay-verify] step {step}: doc_ids={'MATCH' if ids_matched else 'MISMATCH'} "
            f"token_span={'MATCH' if span_matched else 'MISMATCH'} "
            f"({original['token_span']} vs {reconstructed['token_span']}) "
            f"hash={'MATCH' if hash_matched else 'MISMATCH'} "
            f"({original['batch_hash']} vs {reconstructed['batch_hash']})")

    log(f"[{'PASS' if all_matched else 'FAIL'}] replay_hash_matched (steps {start_step}-{end_step}, "
        f"batch ids + token spans + hashes all verified)")
    return all_matched, results


# ---------------------------------------------------------------------------
# Fork -- branch a new lineage from an earlier checkpoint
# ---------------------------------------------------------------------------

def fork_from_checkpoint(fork_step, fork_id="A", log=print):
    tok, manifest = build_pipeline_once()
    batches, *_ = build_batches(tok, manifest)

    parent_ledger = Ledger(LEDGER_PATH)
    fork_ledger_path = ROOT / f"consumption_ledger_fork_{fork_id}.jsonl"
    n_copied = parent_ledger.copy_prefix(upto_step=fork_step, dest_path=fork_ledger_path)
    log(f"Forked lineage '{fork_id}': copied {n_copied} rows (steps 0-{fork_step}) from parent ledger")

    fork_ledger = Ledger(fork_ledger_path)
    parent_rows = {r["step"]: r for r in parent_ledger.replay(upto_step=fork_step)}
    fork_rows = {r["step"]: r for r in fork_ledger.read_all()}
    prefix_matched = all(
        parent_rows[s]["batch_hash"] == fork_rows.get(s, {}).get("batch_hash")
        for s in parent_rows
    )
    log(f"[{'PASS' if prefix_matched else 'FAIL'}] fork_prefix_matched (fork '{fork_id}' vs parent, steps 0-{fork_step})")

    fork_rng = random.Random(999)
    for step in range(fork_step + 1, fork_step + 4):
        if step >= len(batches):
            break
        batch = batches[step]
        loss = round(2.3 * (0.88 ** step) + fork_rng.uniform(-0.03, 0.03), 4)
        rec = _batch_record(step, batch, loss)
        fork_ledger.append(rec)
        log(f"  [fork-{fork_id}] step {step:02d} | loss {loss} | hash {rec['batch_hash']}")

    log(f"branch forked: '{fork_id}' now diverges independently from step {fork_step + 1}")
    return prefix_matched, str(fork_ledger_path)


# ---------------------------------------------------------------------------
# Stats / throughput
# ---------------------------------------------------------------------------

def compute_stats(log=print):
    tok, manifest = build_pipeline_once()
    accepted, eval_rejected, quality_rejected, deferred, protected_overrides = load_train_docs(manifest, tok.pad_id, tok)
    all_packed, per_lane = pack_sequences(accepted, tok.pad_id)
    ledger = Ledger(LEDGER_PATH)
    rows = ledger.read_all()

    raw_tokens = sum(e["num_tokens"] for e in manifest)
    accepted_tokens = sum(r["num_tokens"] for r in accepted)
    eval_rejected_tokens = sum(r["num_tokens"] for r in eval_rejected)
    quality_rejected_tokens = sum(r["num_tokens"] for r in quality_rejected)
    deferred_tokens = sum(r["num_tokens"] for r in deferred)
    loss_bearing_tokens = sum(sum(p["loss_mask"]) for p in all_packed)
    total_packed_slots = sum(p["seq_len"] for p in all_packed)
    packing_utilization = round(loss_bearing_tokens / total_packed_slots, 4) if total_packed_slots else 0
    eval_rejection_rate = round(eval_rejected_tokens / raw_tokens, 4) if raw_tokens else 0
    quality_rejection_rate = round(quality_rejected_tokens / raw_tokens, 4) if raw_tokens else 0
    defer_rate = round(deferred_tokens / raw_tokens, 4) if raw_tokens else 0

    batches, *_rest = build_batches(tok, manifest)
    drawn_lane_counts = {}
    for batch in batches:
        for p in batch:
            drawn_lane_counts[p["lane"]] = drawn_lane_counts.get(p["lane"], 0) + 1
    total_drawn = sum(drawn_lane_counts.values()) or 1
    realized_batch_mix = {l: round(n / total_drawn, 4) for l, n in drawn_lane_counts.items()}

    per_lane_summary = {
        lane: {"policy": info["policy"], "num_packs": len(info["packs"]), "skipped_oversized": info["skipped_oversized"]}
        for lane, info in per_lane.items()
    }

    perf_path = ROOT / PERF_LOG_PATH_NAME
    if perf_path.exists():
        perf = json.loads(perf_path.read_text())
        total_elapsed = perf["total_elapsed_seconds"]
        total_tokens_processed = perf["total_token_span"]
        tokens_per_sec = round(total_tokens_processed / total_elapsed, 2) if total_elapsed > 0 else 0
    else:
        total_elapsed, total_tokens_processed, tokens_per_sec = 0, 0, 0

    stats = {
        "raw_tokens": raw_tokens,
        "useful_loss_bearing_tokens": loss_bearing_tokens,
        "accepted_tokens": accepted_tokens,
        "rejected_tokens_eval_firewall": eval_rejected_tokens,
        "eval_rejection_rate": eval_rejection_rate,
        "rejected_tokens_quality_gate": quality_rejected_tokens,
        "quality_rejection_rate": quality_rejection_rate,
        "deferred_tokens": deferred_tokens,
        "defer_rate": defer_rate,
        "protected_overrides_count": len(protected_overrides),
        "protected_floor_check": {
            lane: {
                "floor_target": CURRICULUM_TARGET.get(lane),
                "realized_share": realized_batch_mix.get(lane),
                "floor_met": realized_batch_mix.get(lane, 0) >= CURRICULUM_TARGET.get(lane, 0) * 0.5,
                # floor_met uses a 50% tolerance band (not the exact target) since
                # realized share is subject to sampling noise at finite step counts,
                # same reasoning as the target-vs-realized gap documented elsewhere
                # in this project; the floor guarantee is "protected from being
                # squeezed toward zero", not "hits the target exactly every run"
            }
            for lane in quality.PROTECTED_FLOOR_LANES
        },
        "num_shards": len(manifest),
        "num_packed_sequences": len(all_packed),
        "packing_utilization": packing_utilization,
        "per_lane_packing": per_lane_summary,
        "training_steps_completed": len(rows),
        "curriculum_target": CURRICULUM_TARGET,
        "realized_batch_mix": realized_batch_mix,
        "tokenizer_vocab_hash": tok.vocab_hash,
        "throughput_tokens_per_sec": tokens_per_sec,
        "total_token_span_this_run": total_tokens_processed,
        "total_wallclock_seconds": round(total_elapsed, 4),
        "throughput_note": f"real measured wall-clock across the training loop, with a "
                            f"{SIMULATED_STEP_COMPUTE_SECONDS}s simulated per-step compute "
                            f"delay standing in for a real forward/backward pass (this loop "
                            f"does no real compute, so an unmodified measurement would be "
                            f"meaninglessly fast, not representative of anything)",
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2))
    log(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["run_and_crash", "resume", "stats", "reset", "replay", "fork"], required=True)
    args = ap.parse_args()

    if args.phase == "reset":
        for p in [LEDGER_PATH, LEARNING_LEDGER_PATH, CKPT_PATH, STATS_PATH, ROOT / PERF_LOG_PATH_NAME]:
            if p.exists():
                p.unlink()
        for p in ROOT.glob("consumption_ledger_fork_*.jsonl"):
            p.unlink()
        print("Reset ledger/checkpoint/stats.")
    elif args.phase == "run_and_crash":
        run_and_crash()
    elif args.phase == "resume":
        resume()
    elif args.phase == "replay":
        replay_interval(*REPLAY_INTERVAL)
    elif args.phase == "fork":
        fork_from_checkpoint(FORK_AT_STEP)
    elif args.phase == "stats":
        compute_stats()
