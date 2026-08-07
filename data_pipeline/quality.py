"""
Quality classifier — the session's "opus" scorer/rejection concept.

Real systems (Setu, FineWeb-Edu-style pipelines) train a lightweight
classifier on labels from a strong LLM judge. We don't have a judge model
here, so this is a HEURISTIC STAND-IN, clearly flagged as such: it scores
punctuation-stripped alpha-word ratio, average clean-word length, and
per-word repetition, and rejects below a threshold. It approximates the
*shape* of a real quality gate (score -> threshold -> reject) without
claiming to be one.

Sanity-checked against both real corpus text and synthetic garbage
(digit/symbol noise, repeated tokens) so the threshold actually
discriminates rather than passing everything -- see
`if __name__ == "__main__"` below for the check.

Protected-flow override: the session transcript specifically asks whether
a lane can override an "opus rejection" so it's never silently dropped
(e.g. Indic data should stay in even if a generic quality classifier scores
it oddly, since script/length statistics differ across languages and a
classifier tuned on English text will under-score non-English prose).
PROTECTED_FLOOR_LANES defines which lanes have a protected floor: OPUS
reject/defer decisions for these lanes are overridden to accept, which
guarantees the lane's realized share can never be squeezed toward zero
by the quality gate. Design choice, stated plainly: this is a full
override (every doc in the lane is protected), not a partial
minimum-share mechanism (override only as many docs as needed to hit a
target percentage). A partial-floor version is a real, more precise
reading of "protected floor" and is a natural next iteration; this
version guarantees the floor is met (in fact exceeds it) rather than
computing the minimum intervention needed to just meet it.
"""
import string
from collections import Counter

REJECT_THRESHOLD = 0.45   # below this: reject
DEFER_THRESHOLD = 0.60    # between REJECT and DEFER: uncertain, deferred for review
PROTECTED_FLOOR_LANES = {"hi"}  # Indic lane has a protected floor per curriculum plan


def score(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0

    # punctuation-stripped alpha-word ratio -- catches digit/symbol noise
    # that char-level alpha ratio misses (a word like "wife." is still a
    # clean word once trailing punctuation is stripped).
    clean_words = [w.strip(string.punctuation) for w in words]
    clean_words = [w for w in clean_words if w]
    if not clean_words:
        return 0.0
    alpha_word_ratio = sum(w.isalpha() for w in clean_words) / len(clean_words)

    avg_len = sum(len(w) for w in clean_words) / len(clean_words)
    # sane prose/code/identifier length is roughly 2-12 chars; garbage
    # tokens tend to be short digit/symbol runs or degenerate repeats
    length_score = 1.0 if 2 <= avg_len <= 14 else max(0.0, 1.0 - abs(avg_len - 8) / 14)

    counts = Counter(clean_words)
    most_common = counts.most_common(1)[0][1] if counts else 0
    repetition_ratio = most_common / len(clean_words)
    repetition_score = 1.0 - min(repetition_ratio, 1.0)

    s = (0.6 * alpha_word_ratio + 0.4 * length_score) * repetition_score
    return round(s, 4)


def evaluate(doc: dict) -> dict:
    """
    OPUS decision: accept / reject / defer, with a protected-lane override.

      score < REJECT_THRESHOLD                -> reject (unless protected)
      REJECT_THRESHOLD <= score < DEFER_THRESHOLD -> defer (uncertain,
          held out of this run rather than trained on OR silently dropped;
          a real system would route deferred docs to human/LLM review)
      score >= DEFER_THRESHOLD                -> accept

    Returns quality metadata stored on the doc/shard record:
      quality_score        -- raw heuristic score, 0..1
      opus_decision         -- "accept" | "reject" | "defer" (pre-override)
      protected             -- is this doc's lane exempt from reject/defer
      opus_decision_final    -- the decision that actually applies after
                                the protected-lane override (always "accept"
                                for protected docs that would've been
                                rejected or deferred)
      quality_reject        -- legacy bool, kept for backward compat
      quality_reject_final  -- legacy bool, kept for backward compat
    """
    s = score(doc["text"])
    protected = doc["lane"] in PROTECTED_FLOOR_LANES

    if s < REJECT_THRESHOLD:
        decision = "reject"
    elif s < DEFER_THRESHOLD:
        decision = "defer"
    else:
        decision = "accept"

    final_decision = "accept" if protected and decision != "accept" else decision

    return {
        "quality_score": s,
        "opus_decision": decision,
        "protected": protected,
        "opus_decision_final": final_decision,
        # legacy fields, kept so older code (packing.py) still works
        "quality_reject": decision == "reject",
        "quality_reject_final": final_decision == "reject",
    }


if __name__ == "__main__":
    # Sanity check: does the classifier actually discriminate, or does it
    # pass everything? Run against real corpus text + synthetic garbage.
    from raw_corpus import generate_corpus

    docs = generate_corpus(seed=42)
    scores = sorted(score(d["text"]) for d in docs)
    print(f"real corpus ({len(scores)} docs): min={scores[0]} p10={scores[len(scores)//10]} "
          f"median={scores[len(scores)//2]} max={scores[-1]}")
    print(f"real docs below threshold ({REJECT_THRESHOLD}): {sum(1 for s in scores if s < REJECT_THRESHOLD)}")

    garbage_samples = [
        "xk3 qz9!! ##$$ 111 222 aaaa aaaa aaaa aaaa",
        "1 2 3 4 5 6 7 8 9 10 11 12 13 14",
        "asdf asdf asdf asdf asdf asdf asdf asdf",
    ]
    for g in garbage_samples:
        s = score(g)
        print(f"garbage {g!r}: score={s} reject={s < REJECT_THRESHOLD}")
