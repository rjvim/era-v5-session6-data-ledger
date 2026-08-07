"""
Character-level BPE with an explicit byte-fallback layer, and a
code-aware pre-tokenization pattern that fixed a fertility regression
this project's own tokenizer went through -- documented honestly rather
than only showing the final answer.

History, in order:
  1. Whole-word dict lookup: could produce total <unk> on any unseen
     word ("शिवाय" -> 100% unknown, despite "शिव" being known elsewhere).
  2. Byte-level BPE: fixed unknowns completely, but silently taxed
     Devanagari. UTF-8 encodes Hindi characters as 2-3 bytes vs 1 for
     English, so most of a small vocab's merge budget went to just
     reconstructing characters from bytes before reaching word-level
     merges.  Measured: en fertility 1.36, hi fertility 3.31.
  3. Tried the "obvious" fixes -- bigger vocab (up to 48k), oversampling
     Hindi (up to 50x) -- both PLATEAUED completely flat around hi=3.27.
     That flatness was the evidence the bottleneck was structural
     (byte-vs-character), not a frequency-competition problem.
  4. Switched the base alphabet from bytes to Unicode characters
     (Metaspace pre-tokenizer). Fixed en/hi fertility (1.19 / 1.26), but
     (a) silently dropped whitespace on decode -- a real bug, caught by
     testing round-trip equality, not assumed -- and (b) made `code`
     lane fertility WORSE (1.72 -> 3.08), because Metaspace turns every
     single space in a run into its own separate token: a 4-space
     indent became 4 wasted tokens instead of 1.
  5. This version: a regex pre-tokenizer that (a) collapses runs of 2+
     whitespace into ONE token -- fixes the indentation blowup -- while
     (b) still fusing a single leading space into the following word,
     preserving the efficiency that made prose fertility good. Result:
     en 1.29, hi 1.32, code 2.21 -- all three lanes reasonably close,
     with correct, simple round-trip (tokens carry literal text, decode
     is plain concatenation, no decoder object that can silently
     misbehave).
  6. Pushed code fertility further: traced the actual token output for
     a real Flask function signature and found multi-char operators
     splitting wastefully -- "->" became two tokens (' -' + '>'), and a
     Python docstring's opening triple-quote became three separate single
     quote tokens. Fixed by grouping RUNS of punctuation into one
     pretoken instead of one-at-a-time. Then re-tested vocab_size, which
     had plateaued uselessly in step 3 for the byte-tax reason -- with
     that cause no longer applying, it now genuinely helped: en
     1.29->1.17, hi 1.32->1.12, code 2.21->1.8. Confirmed a real ceiling
     (not guessed): requesting vocab_size above ~44k has zero further
     effect on this corpus, verified by testing up to 96k.

Byte-fallback (unchanged in spirit since step 2): any character the
trained vocab doesn't cover -- a script genuinely absent from this
corpus, e.g. Japanese or emoji -- gets encoded as its raw UTF-8 bytes in
a reserved id range, so this tokenizer can never truly fail to encode
something, the way the original whole-word version could.
"""
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Split

PAD, EOS, UNK = "<pad>", "<eos>", "[UNK]"
SPECIAL_TOKENS = [PAD, EOS, UNK]
VOCAB_SIZE = 64000  # requesting more has no effect: this corpus's natural
                     # merge ceiling is ~49,506 tokens (confirmed by testing
                     # up to 200,000 and seeing zero further change) --
                     # raised from ~44,173 after adding 5.7x more real code
                     # data (Click/Jinja2/Requests/Werkzeug alongside Flask)
BYTE_FALLBACK_BASE = 100000  # reserved id range for raw-byte fallback tokens

# Priority order matters (first alternative match wins per position):
#   1. \s{2,}   -- runs of 2+ whitespace collapse to ONE token (fixes code
#                  indentation: a 4-space indent is 1 token, not 4)
#   2. \s?\w+   -- an optional single leading space fuses into the
#                  following word (keeps normal prose efficient, same
#                  idea as GPT-2/Metaspace's "space belongs to the next
#                  word" convention, but without Metaspace's per-space
#                  token blowup on indentation)
#   3. \s?[^\s\w]+ -- an optional single leading space fuses into a RUN
#                  of punctuation/symbol characters (not just one), so
#                  multi-char operators like "->", "==", or a triple-quote
#                  """ become one pretoken instead of splitting into 2-3
#                  wasted single-symbol tokens. This specifically fixed a
#                  measured code-fertility gap: a real Flask function
#                  signature was tokenizing "->" as two separate tokens
#                  (' -' + '>') and '"""' as three, confirmed by inspecting
#                  the actual token trace before changing anything.
PRETOKEN_PATTERN = Regex(r"\s{2,}|\s?\w+|\s?[^\s\w]+")


def build_vocab(corpus_lines, vocab_path):
    """Train character-level BPE once on the full real corpus, freeze to disk."""
    tok = Tokenizer(BPE(unk_token=UNK))
    tok.pre_tokenizer = Split(PRETOKEN_PATTERN, behavior="isolated")
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)
    tok.train_from_iterator(corpus_lines, trainer)
    tok.save(str(vocab_path))
    return tok


class FrozenTokenizer:
    def __init__(self, vocab_path):
        self._tok = Tokenizer.from_file(str(vocab_path))
        self.pad_id = self._tok.token_to_id(PAD)
        self.eos_id = self._tok.token_to_id(EOS)
        self.unk_id = self._tok.token_to_id(UNK)

        raw = Path(vocab_path).read_text()
        self.vocab_hash = hashlib.sha256(raw.encode()).hexdigest()[:12]

    def encode(self, text):
        """
        Encode with character-level BPE, then explicitly byte-fallback any
        span the trained vocab doesn't cover -- keeps the same zero-unknown
        guarantee as byte-level BPE, without its fertility cost on scripts
        the vocab DOES cover well.
        """
        enc = self._tok.encode(text, add_special_tokens=False)
        ids = []
        for tid, (start, end) in zip(enc.ids, enc.offsets):
            if tid == self.unk_id and end > start:
                span = text[start:end]
                ids.extend(BYTE_FALLBACK_BASE + b for b in span.encode("utf-8"))
            else:
                ids.append(tid)
        ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        """
        Tokens from this pre-tokenizer carry their literal text (including
        any fused leading space) verbatim, so decoding is plain
        concatenation -- no decoder object needed, no risk of a decoder
        silently dropping or adding whitespace.
        """
        ids = [i for i in ids if i != self.eos_id]
        pieces = []
        byte_buf = []

        def flush_bytes():
            if byte_buf:
                pieces.append(bytes(byte_buf).decode("utf-8", errors="replace"))
                byte_buf.clear()

        for i in ids:
            if i >= BYTE_FALLBACK_BASE:
                byte_buf.append(i - BYTE_FALLBACK_BASE)
            else:
                flush_bytes()
                pieces.append(self._tok.id_to_token(i) or "")
        flush_bytes()
        return "".join(pieces)
