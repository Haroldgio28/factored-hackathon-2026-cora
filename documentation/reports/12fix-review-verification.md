# Verification note — 12 semantic-review fixes (branch `feat/agent-orchestration`)

All commands run from `c:\Users\1872146\Documents\CORA` with `CORA_NLU_STUB=1`. Nothing staged or
committed; the working tree is left modified for the commit step.

## Commands run and results

- `uv run ruff check .` → **All checks passed!**
- `uv run ruff format --check .` → **140 files already formatted**
- `uv run pytest` → **364 passed, 1 skipped, 20 warnings** (the skip and the sklearn deprecation
  warnings pre-date this change). Per-file counts for the touched suites: `test_generator.py` 13,
  `test_templates.py` 14, `test_grounding.py` 16, `test_llm.py` 4 (new).
- `uv run python -m cora.agent.generator` → `generator stub self-check OK` (exit 0). The self-check
  asserts the EXACT review phrasings (see below).
- Demo `scripts/aws/demo_agent_bedrock.py` was NOT run (needs a live Bedrock bearer token). Verified
  by import + lint: it imports cleanly and `_REPORT_PATH` resolves to
  `documentation/reports/demo_agent_bedrock.txt`.

## Language-guard correctness demonstration

`_flipped_language(text, draft_lang)` returns the matched opposite-language marker or `None`.
Demonstrated in the generator `__main__` self-check AND in `tests/unit/test_generator.py`:

Ordinary Spanish that MUST NOT flip (draft_lang="es", issue #1) — all return `None`:
- "Tu tarjeta está activa." · "Esa información no está disponible por este canal." ·
  "Reporté el robo de tu tarjeta SIM." · "Lo consigo por este canal sin problema." ·
  "No puedo ayudarte con eso, gracias por tu comprensión."

pt→es translations the demo produced that MUST be caught (draft_lang="pt", issue #2) — all non-None:
- "Su saldo actual es de 1.250,75 COP y tiene 1.100,00 COP disponibles para usar" → matches `su`
- "Voy a transferir tu caso a una persona del equipo" → matches `Voy`

Faithful Portuguese polish (draft_lang="pt") returns `None`:
- "Vou encaminhar o seu caso a uma pessoa da equipe"

Non-es/pt early return (draft_lang="en") returns `None`.

## Issue-by-issue

1/2. **Guard redesigned** (`src/cora/agent/generator.py`): keys only on language-EXCLUSIVE signals.
   pt by orthography (`ã õ ç nh lh` + `não você cartão então` + standalone `é`); es by
   accent-independent exclusive function words (`voy una persona equipo disponibles su dinero
   "es de" usted(es)`), because a translated balance/escalation answer has no `ñ ¿ ¡`. Shared tokens
   (`está consigo sim voces`) removed. The asymmetry is documented honestly in the module comment;
   the English prompt is the PRIMARY fix, the guard the deterministic backstop.
3. **Precision pinned**: table-driven negative cases (`test_ordinary_spanish_is_never_read_as_a_flip`),
   the pt→es demo-shape positives (`test_pt_to_es_translations_in_demo_shapes_are_caught`), and the
   non-es/pt early return (`test_non_es_pt_language_returns_no_flip`).
4. **Demo section 3 renamed** to an honest figure-free smoke test with an honest conclusion (the
   grounding rejection branch is covered by `tests/unit/test_grounding.py`).
5. **`language_blocked` surfaced** in the demo `_show` next to `grounding_blocked`.
6. **Converse empty-system test** added (`tests/unit/test_llm.py`): a fake `converse` captures
   kwargs; `system` absent for "" and whitespace, present with `[{"text": ...}]` otherwise; model id
   from settings only. No network.
7. **Both gates evaluated, both flags set** (`generator.py`): grounding and language are no longer
   order-dependent; each unsafe outcome is counted by type.
8. **Demo honest-evidence guard**: requires ≥1 polished turn (else exit non-zero), prints the model
   id, writes the transcript to `documentation/reports/`, annotates `response: GeneratedResponse`.
9. **Docs updated**: `tasks.md` 4.2/4.3 (English prompt, new guard, new counts, prompt hash
   `1b90b2a98808…`) and the REQ-18 row (SRC-45) in `TRACEABILITY.md` (generator guard + demo script).
10. **AWS profile**: demo now constructs `BedrockLLMClient(Settings(aws_profile=""))` so the
    bearer-token path is used (the `os.environ.pop` was a no-op because the client reads the profile
    from `Settings`, and `get_settings` is `lru_cache`d).
11. **`lang` normalized once** (`_normalize_lang`) before the template lookup and the prompt
    interpolation; test `test_unsupported_language_normalizes_to_es_and_guard_is_silent`.
12. **Warning log enriched**: includes `lang` and the matched marker, matching the grounding
    warning's shape in the same module.
