# Evidence summary

| Requirement | Result | Evidence |
|---|---|---|
| Tokenizer integrity | PASS | Manifest record (tokenizer_vocab_hash on every shard) |
| Evaluation firewall | PASS | Blocked-shard event (this run.log line) |
| Packing correctness | PASS | Packed-batch report (performance.json) |
| OPUS audit trail | PASS | Candidate decision records (opus_decision field on every shard record) |
| Crash recovery | PASS | Expected and resumed batch ids (run.log) |
| Replay | PASS | Original and replay hashes (run.log) |
| Learning trace | PASS | Loss linked to source data (learning_ledger.jsonl) |
| Throughput | PASS | Performance report (performance.json) |
| Mixture compliance | PASS | Planned versus actual shares (performance.json) |
