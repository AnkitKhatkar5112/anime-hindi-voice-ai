# Phase 4 — Hindi Pipeline Refinement: Tasks

- [x] 1. Language config abstraction validation
  - [x] 1.1 Validate that adding a new language requires only `configs/languages.yaml` edits
    - Document any code change that was required (there should be none)
    - If a code change was needed, fix the abstraction
    - **Done when:** Pipeline runs (or fails gracefully) with zero Python code changes for a new config entry
    - _Requirements: 1_
  - [x] 1.2 Write `docs/adding_a_language.md` step-by-step guide
    - Document the YAML config fields, model availability check, dialect script wiring
    - **Done when:** `docs/adding_a_language.md` exists with clear instructions
    - _Requirements: 1_
