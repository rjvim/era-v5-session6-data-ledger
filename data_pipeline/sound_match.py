"""
Cross-script / cross-spelling phrase matcher.

Scope, deliberately limited: this is a MATCHING utility, not a change to
how the pipeline tokenizes or trains on text. The `hi` lane's real
Devanagari training data is untouched -- the tokenizer still sees and
learns real Devanagari, so the model can still read and write actual
Hindi/Sanskrit script, not just a romanized stand-in. Folding all Hindi
training data down to romanized "sound form" would have destroyed that
capability entirely, which defeats the point of having a Hindi lane.

What this DOES do: given two strings -- any mix of Devanagari script,
casual Latin transliteration, or different casing/spelling -- decide
whether they're plausibly the same underlying phrase. Useful for search,
deduplication-across-script, or "did the user already ask this" checks,
without altering any training data.

Two layers, same idea tested earlier in this project:
  1. Devanagari -> IAST transliteration (via the indic_transliteration
     library) -> strip diacritics -> lowercase. This is the "sound form".
     Latin input skips straight to the fold step.
  2. Fuzzy similarity (difflib) on the sound-formed strings, not exact
     equality -- because casual spelling ("sivaya" vs "shivaya") doesn't
     perfectly converge even after sound-folding, as found earlier.
     Measured similarity across real spelling variants of "om namah
     shivaya" landed between 0.93 and 1.00; MATCH_THRESHOLD is set below
     that range.
"""
import difflib
import re
import unicodedata

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F]")
MATCH_THRESHOLD = 0.85


def to_sound_form(text: str) -> str:
    """Normalize Devanagari or Latin text to a comparable romanized form."""
    if DEVANAGARI_RANGE.search(text):
        text = transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    ascii_str = ascii_str.replace("ś", "s").replace("ṣ", "s").replace("ñ", "n")
    return re.sub(r"\s+", " ", ascii_str.lower()).strip()


def similarity(a: str, b: str) -> float:
    """0..1 similarity between two strings after sound-form normalization."""
    return difflib.SequenceMatcher(None, to_sound_form(a), to_sound_form(b)).ratio()


def matches(a: str, b: str, threshold: float = MATCH_THRESHOLD) -> bool:
    """Are these plausibly the same phrase, regardless of script/spelling?"""
    return similarity(a, b) >= threshold


if __name__ == "__main__":
    # sanity check against every variant tested earlier in this project
    reference = "ॐ नमः शिवाय"
    variants = ["om namah shivaya", "om nama shivaya", "om namah sivaya",
                "Om Namah Shivaya", "aum namah shivaya"]
    for v in variants:
        s = similarity(v, reference)
        print(f"{v!r:22s} vs {reference!r}: similarity={s:.2f}  match={matches(v, reference)}")

    # negative control -- should NOT match
    unrelated = "the cat sat on the mat"
    s = similarity(unrelated, reference)
    print(f"{unrelated!r:22s} vs {reference!r}: similarity={s:.2f}  match={matches(unrelated, reference)}  (expect False)")
