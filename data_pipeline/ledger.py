"""
The DATA LEDGER.

Every training step, before we "train" on a batch, we APPEND a record
of exactly what was consumed and in what order. This is separate from
the training ledger (hyperparams/model state) — this one only tracks
data consumption order.

Why: if we crash and resume, we do NOT recompute or re-sample data.
We READ the ledger to know exactly what already happened, replay it
if needed, and continue from the next unwritten step. This removes
the nondeterminism risk of re-running sampling code.
"""
import json
from pathlib import Path


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            self.path.write_text("")

    def append(self, record):
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def read_all(self):
        if not self.path.exists():
            return []
        lines = self.path.read_text().splitlines()
        return [json.loads(l) for l in lines if l.strip()]

    def last_step(self):
        rows = self.read_all()
        return rows[-1]["step"] if rows else -1

    def replay(self, upto_step=None):
        """Return the recorded batches in original order, no recomputation."""
        rows = self.read_all()
        if upto_step is not None:
            rows = [r for r in rows if r["step"] <= upto_step]
        return rows

    def copy_prefix(self, upto_step, dest_path):
        """Write the ledger's rows up to and including upto_step into a new
        file at dest_path -- used to fork a lineage from a checkpoint: the
        fork starts as an exact copy of the parent up to the fork point,
        then diverges independently from there."""
        rows = self.replay(upto_step=upto_step)
        dest = Path(dest_path)
        dest.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))
        return len(rows)


class Checkpoint:
    """Stores enough to resume AND to fork: step number, RNG state, and
    the exact ledger offset (line count) it corresponds to -- so a
    checkpoint is always tied to a precise, verifiable position in the
    ledger, not just a step number that could drift out of sync."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)  # don't rely on an
        # empty directory surviving git/upload -- empty dirs aren't tracked by
        # git and can silently vanish (e.g. GitHub's web upload drops them),
        # so create it ourselves rather than assuming it exists

    def save(self, step, rng_state, ledger_offset):
        payload = {"step": step, "rng_state": rng_state, "ledger_offset": ledger_offset}
        self.path.write_text(json.dumps(payload))

    def load(self):
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text())
