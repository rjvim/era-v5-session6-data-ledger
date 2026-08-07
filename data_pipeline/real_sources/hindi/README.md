# hi lane — real source

**Source:** `ai4bharat/sangraha`, `verified` config, `hin` split (Hindi,
Devanagari). Verified = human-origin, not machine-translated/synthetic.
Pulled via the `datasets-server.huggingface.co` REST API (`/rows`
endpoint), since bare `huggingface.co` — including `datasets.load_dataset()`
and `hf_hub_download()` — is still blocked by this sandbox's egress
allowlist (`host_not_allowed`). `*.huggingface.co` subdomains are allowed;
the apex domain is not.

**File:** `sangraha_verified_hin.jsonl` — 300 documents, each `{doc_id, text}`,
filtered to >200 chars, doc order as returned by the API (offset-paginated,
100 rows/page, first ~400 rows scanned).

**License:** CC-BY-4.0 (Sangraha).

**Replaces:** the authored Sanskrit placeholder previously in this lane
(GRETIL Upanishad text — real but not modern Hindi, used only because
`huggingface.co` was unreachable at the time).
