"""Agent package (Phase 3+): the one LLM interface the learned-NLU and Phase-5 agent share.

Phase 3 adds `llm.py` only. The LLM proposes; deterministic code decides (principle P1): the
zero-shot baseline (3.4) and entity extraction (3.7) call `LLMClient` and treat its output as an
untrusted suggestion that is validated and fails closed, never as a decision.
"""
