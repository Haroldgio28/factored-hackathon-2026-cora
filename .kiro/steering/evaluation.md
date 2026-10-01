---
inclusion: fileMatch
fileMatchPattern: "{src/cora/eval/**,src/cora/nlu/**,analysis/**,documentation/reports/**}"
---

# Evaluation & evidence rules

- Compare baselines (B0 historical, B1 naive LLM, intent baselines) and CORA on the **same held-out workload**.
- Splits are grouped by `customer_id` and ordered in time; label-bearing fields are never model inputs.
- Report with numerators/denominators and 95% Wilson CIs; zero unsafe events -> rule-of-three upper bound.
- Run each configuration ≥3 times; record model ids, prompt hashes, code commit and data snapshot.
- Metrics: safe automated resolution (+ attempted share), containment, escalation precision/recall (missed and unnecessary), unsafe outcomes by type, p50/p95 latency, cost per attempted case and per SAR ("not defined" if none).
- Break down by language, country/accent and segment; investigate gaps.
- Label every number as offline measurement, simulation or projection. Include failures and error analysis.
- Deterministic judges first; an LLM judge only for subjective qualities, with a published rubric validated on ≥50 human labels.
- Never tune thresholds on the test split.
