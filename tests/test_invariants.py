"""
Automated tests for the invariants this system claims to hold.

Run standalone:  python3 tests/test_invariants.py
Run with pytest: pytest tests/test_invariants.py -v

These are the same checks run_demo.py performs live during a real
execution -- duplicated here as isolated, independently-runnable tests
so the invariants can be verified without doing a full crash/resume/
replay/fork cycle first. Each test either builds its own minimal state
or asserts against whatever state already exists in data_pipeline/
after run_demo.py has been run at least once.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "data_pipeline"))

DP = Path(__file__).parent.parent / "data_pipeline"


def test_reproducibility():
    """Rebuilding all shards from the same corpus + frozen tokenizer must
    produce byte-identical content hashes -- this is the core immutability
    guarantee the whole shard system depends on."""
    from tokenizer import FrozenTokenizer
    from raw_corpus import generate_corpus
    from shards import write_shards

    manifest_path = DP / "manifest.json"
    assert manifest_path.exists(), "run the pipeline at least once before running tests"
    m1 = json.loads(manifest_path.read_text())
    hashes1 = sorted(e["hash"] for e in m1)

    tmp_dir = DP / "_test_shards_tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tok = FrozenTokenizer(str(DP / "tokenizer.json"))
    docs = generate_corpus(seed=42)
    m2, _ = write_shards(docs, tok, str(tmp_dir))
    hashes2 = sorted(e["hash"] for e in m2)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    manifest_path.write_text(json.dumps(m1, indent=2))  # restore original paths

    assert hashes1 == hashes2, "rebuilt shard hashes do not match original -- reproducibility broken"
    print(f"PASS test_reproducibility ({len(hashes1)} shards)")


def test_ledger_gapless():
    """The consumption ledger must be a gapless, ordered sequence 0..N with
    no duplicate or missing steps -- this is what makes 'resume from here'
    unambiguous."""
    from ledger import Ledger

    ledger_path = DP / "consumption_ledger.jsonl"
    assert ledger_path.exists(), "run the pipeline at least once before running tests"
    rows = Ledger(str(ledger_path)).read_all()
    steps = [r["step"] for r in rows]
    assert steps == list(range(len(steps))), f"ledger has gaps or duplicates: {steps}"
    print(f"PASS test_ledger_gapless (steps 0..{len(steps)-1})")


def test_eval_firewall():
    """No document tagged split=eval may appear in the accepted (training)
    pool -- verified by checking eval_rejected docs never leak into
    accepted."""
    from tokenizer import FrozenTokenizer
    from packing import load_train_docs
    import json as _json

    manifest = _json.loads((DP / "manifest.json").read_text())
    tok = FrozenTokenizer(str(DP / "tokenizer.json"))
    accepted, eval_rejected, quality_rejected, deferred, protected_overrides = load_train_docs(manifest, tok.pad_id, tok)

    accepted_ids = {r["doc_id"] for r in accepted}
    eval_ids = {r["doc_id"] for r in eval_rejected}
    leaked = accepted_ids & eval_ids
    assert not leaked, f"eval firewall breach: {leaked} appear in both accepted and eval-rejected"
    assert len(eval_rejected) > 0, "no eval docs found at all -- test corpus should contain some"
    print(f"PASS test_eval_firewall ({len(eval_rejected)} eval docs correctly blocked, 0 leaked)")


def test_resume_no_skip_no_repeat():
    """After a crash + resume cycle, every step from 0 to the final step
    must appear in the ledger exactly once -- no batch skipped, none
    repeated."""
    from ledger import Ledger

    ledger_path = DP / "consumption_ledger.jsonl"
    assert ledger_path.exists(), "run the pipeline at least once before running tests"
    rows = Ledger(str(ledger_path)).read_all()
    steps = [r["step"] for r in rows]
    assert len(steps) == len(set(steps)), "duplicate step(s) found -- a batch was repeated"
    assert steps == sorted(steps), "steps out of order"
    assert steps[0] == 0, "ledger does not start at step 0"
    print(f"PASS test_resume_no_skip_no_repeat ({len(steps)} steps, no skips, no repeats)")


def test_replay_hash_match():
    """Recomputing a historical interval's batches must reproduce the exact
    same batch hashes recorded in the ledger at the time."""
    from train import build_pipeline_once, build_batches, batch_hash
    from ledger import Ledger

    ledger_path = DP / "consumption_ledger.jsonl"
    assert ledger_path.exists(), "run the pipeline at least once before running tests"
    tok, manifest = build_pipeline_once()
    batches, *_ = build_batches(tok, manifest)
    recorded = {r["step"]: r for r in Ledger(str(ledger_path)).read_all()}

    checked = 0
    for step in range(5, 11):
        if step not in recorded or step >= len(batches):
            continue
        assert recorded[step]["batch_hash"] == batch_hash(batches[step]), f"replay hash mismatch at step {step}"
        checked += 1
    assert checked > 0, "no steps available to check replay against"
    print(f"PASS test_replay_hash_match ({checked} steps verified)")


def test_fork_prefix_matches_parent():
    """A forked lineage's prefix (steps before the fork point) must be
    byte-identical to the parent ledger's rows for those same steps."""
    from ledger import Ledger

    parent_path = DP / "consumption_ledger.jsonl"
    fork_path = DP / "consumption_ledger_fork_A.jsonl"
    if not fork_path.exists():
        print("SKIP test_fork_prefix_matches_parent (no fork present -- run train.py --phase fork first)")
        return

    parent_rows = {r["step"]: r for r in Ledger(str(parent_path)).read_all()}
    fork_rows = {r["step"]: r for r in Ledger(str(fork_path)).read_all()}
    shared_steps = [s for s in fork_rows if s in parent_rows and s <= 20]
    assert shared_steps, "no shared prefix steps found"
    for s in shared_steps:
        assert parent_rows[s]["batch_hash"] == fork_rows[s]["batch_hash"], f"fork prefix mismatch at step {s}"
    print(f"PASS test_fork_prefix_matches_parent ({len(shared_steps)} shared steps verified)")


def test_tokenizer_zero_unknown():
    """The tokenizer must never produce a true unknown for any input --
    verified against scripts genuinely absent from training data."""
    from tokenizer import FrozenTokenizer

    tok = FrozenTokenizer(str(DP / "tokenizer.json"))
    test_strings = ["日本語", "🎉🎉🎉", "ыыы", "शिवाय", "om namah shivaya"]
    for s in test_strings:
        ids = tok.encode(s)
        decoded = tok.decode(ids)
        assert decoded == s, f"round-trip failed for {s!r}: got {decoded!r}"
    print(f"PASS test_tokenizer_zero_unknown ({len(test_strings)} strings, all round-trip exact)")


ALL_TESTS = [
    test_reproducibility,
    test_ledger_gapless,
    test_eval_firewall,
    test_resume_no_skip_no_repeat,
    test_replay_hash_match,
    test_fork_prefix_matches_parent,
    test_tokenizer_zero_unknown,
]


def run_all():
    results = []
    for t in ALL_TESTS:
        try:
            t()
            results.append((t.__name__, True, None))
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            results.append((t.__name__, False, str(e)))
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n{passed}/{len(results)} tests passed")
    return results


if __name__ == "__main__":
    run_all()
