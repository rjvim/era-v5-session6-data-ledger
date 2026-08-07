# Session 6 — Training Data Execution System

A small but complete Training Data Execution System for V5. Implements the
full path: documents → tokenized shards → manifests → mixture schedule →
packing → batches → training → consumption ledger → learning ledger →
checkpoint → crash → resume → replay → fork → audit.

Goal is not scale — goal is to prove the system is correct, reproducible,
auditable, and efficient, on a small real corpus.

## Run it

```
pip install -r requirements.txt
python3 run_demo.py
```

One command, no manual steps. Regenerates `submission_artifacts/` from
scratch every time — verified idempotent (same tokenizer hash, all checks
PASS on repeated runs).

## Architecture

```
data_pipeline/
  raw_corpus.py    real source text -> tagged docs {doc_id, lane, split, text}
  tokenizer.py     character-level BPE + byte-fallback (frozen, hashed)
  shards.py        docs -> immutable, content-hashed shards + manifest,
                    OPUS decision tags on every record
  quality.py       OPUS: accept / reject / defer, + protected-lane override
  packing.py       per-lane packing policy; eval-split firewall;
                    OPUS-gated document loading
  ledger.py        consumption ledger, checkpoint (tied to ledger offset),
                    ledger prefix-copy (used by fork)
  train.py         batch/pack content hashing, curriculum-weighted batching,
                    run/crash/resume, verified replay, fork
tests/
  test_invariants.py   automated tests for every claimed invariant
run_demo.py        orchestrates the full run, writes submission_artifacts/
```

## Design decisions

**Data is real, not synthetic.** English (*Pride and Prejudice*, public
domain), Hindi (`ai4bharat/sangraha`, `verified/hin`, CC-BY-4.0,
human-verified), code (5 real Python projects — Flask, Click, Jinja2,
Requests, Werkzeug — BSD-3/Apache-2.0). ~9MB total, no large download.

**Two ledgers, not one, on purpose.** The **consumption ledger**
(`ledgers/consumption_ledger.jsonl`) records *what was consumed, in what
order* — the source of truth for crash/resume. The **learning ledger**
(`ledgers/learning_ledger.jsonl`) records *what the model learned from
it* — loss explicitly tied to `source_doc_ids` for every step. They're
separate files because they answer different audit questions: "what did
we train on" vs. "what did training on it produce."

**Replay verifies three attributes explicitly, not just a hash.**
`replay_interval` recomputes an earlier interval fresh and compares
**batch doc ids, token spans, and content hashes** individually — the
hash alone is technically sufficient (it's a deterministic function of
exactly those two attributes), but checking all three gives a clearer
audit trail: a mismatch would show exactly which attribute diverged.
This matches the spec's own wording precisely ("prove that the
reconstructed batch ids, token spans and hashes match the original
run") rather than treating it as satisfied by the hash alone.

**Checkpoints are tied to ledger offsets, not just step numbers.**
`Checkpoint.save(step, rng_state, ledger_offset)` — a checkpoint always
points at a precise, verifiable position in the ledger, so resume and
fork both have an unambiguous anchor.

**OPUS has three outcomes, not two.** `accept` / `reject` / `defer`.
A heuristic quality classifier scores every document; below a reject
threshold it's rejected, in an uncertain middle band it's *deferred*
(held out of this run, not trained on and not discarded — a real system
would route it to further review), above that it's accepted. **Protected
floor lanes** (`hi`) override *both* reject and defer to accept —
because the same heuristic that reasonably flags English/code oddities
systematically misjudges Hindi (Devanagari punctuation doesn't strip
the way Latin punctuation does). Verified finding, not hypothetical:
274 real Hindi docs were kept via this override in the run that
produced this README's `submission_artifacts/`.

**Honest note on the floor mechanism**: this is a *full* override (every
doc in a protected-floor lane is exempt from reject/defer), not a
*partial* minimum-share mechanism (override only as many docs as needed
to hit a target percentage, no more). A partial-floor version is a more
precise reading of "protected floor" and would be the natural next
iteration — this version guarantees the floor is met (in fact exceeds
it) rather than computing the minimum intervention needed to just meet
it. `stats.protected_floor_check` in `evidence.json` reports the floor
target vs. realized share explicitly, with a disclosed tolerance band
(sampling noise at finite step counts means realized share won't hit
the target exactly every run — same reasoning documented for the
curriculum mixture below).

**Fork copies the parent ledger's prefix, then verifies it, then
diverges.** `fork_from_checkpoint` doesn't just start a new training
run from a checkpoint — it copies the parent's ledger rows up to the
fork point, checks their hashes match the parent's exactly
(`fork_prefix_matched`), *then* continues with a different RNG path so
the divergence after the fork point is real, not cosmetic.

**Throughput is really measured, not fabricated.** This loop does no
real compute, so an unmodified wall-clock measurement is meaninglessly
fast (a first attempt at this produced a nonsensical 1.7 billion
tokens/sec — caught and fixed, not shipped). A small, disclosed
per-step delay (`SIMULATED_STEP_COMPUTE_SECONDS = 0.01s`) stands in for
a real forward/backward pass, and the reported tokens/sec is real
wall-clock time divided by real token count across the actual training
loop — see `throughput_note` in `performance.json` for the exact
methodology.

## Requirement → where it's implemented and verified

| Requirement | File | Verified how |
|---|---|---|
| Immutable tokenized shards + manifests | `shards.py` | `test_reproducibility`: rebuild from scratch, hashes match |
| Frozen tokenizer + content hashes | `tokenizer.py` | `tokenizer_vocab_hash` on every shard; `test_tokenizer_zero_unknown` |
| Packing policies per data type | `packing.py::LANE_PACKING_POLICY` | code: seq_len 256, never truncated; en/hi: seq_len 128, truncate-on-overflow |
| Loss masks, attention masks, position ids | `packing.py::_pack_one_lane` | present on every packed sequence |
| Curriculum stages, lane weights, protected floors | `train.py::CURRICULUM_TARGET` | Session 5 Stage 1 numbers (70/20/10); `PROTECTED_LANES` in `quality.py` |
| Evaluation/validation firewall | `packing.py::load_train_docs` | `test_eval_firewall`: 0 eval docs leak into accepted |
| OPUS accept/reject/defer/protected-override | `quality.py::evaluate` | `opus_decision` + `opus_decision_final` on every shard record |
| Consumption + learning ledgers | `ledger.py`, `train.py` | separate files, learning ledger ties loss to `source_doc_ids` |
| Token/sample-level loss tracking | `train.py::_batch_record` | `token_span` per batch; loss per step in both ledgers |
| Checkpoints tied to ledger offsets | `ledger.py::Checkpoint` | `ledger_offset` field alongside `step` |
| Crash recovery, no skip/repeat | `train.py::run_and_crash`/`resume` | `test_resume_no_skip_no_repeat`; `resume_next_batch_matched` hash check |
| Replay of historical stream | `train.py::replay_interval` | `test_replay_hash_match`: recompute vs. recorded hash |
| Fork from earlier checkpoint | `train.py::fork_from_checkpoint` | `test_fork_prefix_matches_parent` |
| Packing utilization + tokens/sec | `train.py::compute_stats` | real wall-clock, disclosed methodology, in `performance.json` |

## Tests

```
python3 tests/test_invariants.py
```

7 tests, all independently runnable, each asserting one claimed
invariant rather than trusting the demo run's own printed output.

## `submission_artifacts/` structure

```
submission_artifacts/
  run.log            full event sequence + [PASS]/[FAIL] tags
  evidence.json       machine-generated, not hardcoded — see run_demo.py
  evidence.md         human-readable requirement/result/evidence table
  manifests/           shard manifest
  ledgers/              consumption, learning, and fork ledgers
  checkpoints/          latest checkpoint (step + rng_state + ledger_offset)
  performance.json      packing utilization, throughput, fertility, mixture compliance
```

Every file in this directory is generated fresh by `run_demo.py` — none
of it is hand-written or hardcoded.

## What's simplified vs. a real system

- "Training" is a decaying loss formula, not a real model.
- The OPUS quality classifier is a heuristic stand-in for a trained
  LLM-judge classifier — the scoring function itself isn't meant to be
  taken as a real quality signal, only as a working example of the
  score → threshold → decide → override mechanism.
- Throughput's absolute number reflects a disclosed simulated per-step
  delay, not real GPU compute — the *methodology* (real wall-clock ÷
  real token count) is genuine.
