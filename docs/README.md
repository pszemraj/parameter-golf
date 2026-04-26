# Docs

This directory is the committed narrative trail for the local Core/Amplifier work.

Use it for:
- chronological experiment notes with timestamps and commit hashes
- architecture observations that should survive beyond W&B panels
- links back to local reports under `experiments/*/report.md`
- links to important W&B runs or groups

Current conventions:
- `docs/5090_next_experiments.md`
  - short current-state summary and exact next commands
- `docs/5090_final_week_plan.md`
  - maintained closeout protocol for the active 5090 path
- `docs/5090_shape_reassessment.md`
  - current geometry-frontier rationale and promotion rules
- `docs/5090_log.md`
  - append-only logbook for the local RTX 5090 work
  - each entry should include:
    - timestamp
    - commit hash
    - experiment family
    - exact commands or sweep root
    - important W&B links
    - main observations
    - next action
- `experiments/*/report.md`
  - per-family reports backed by completed runs on disk

Raw run artifacts remain local under `experiments/` and in W&B. The committed docs should explain what mattered and why.
