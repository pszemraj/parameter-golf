# Docs

This directory is the committed narrative trail for the local Core/Amplifier work.

Use it for:
- chronological experiment notes with timestamps and commit hashes
- architecture observations that should survive beyond W&B panels
- links back to local reports under `experiments/*/report.md`
- links to important W&B runs or groups

Current conventions:
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
- `notes/`
  - point-in-time summary notes and recommendations
- `experiments/*/report.md`
  - per-family reports backed by completed runs on disk

Raw run artifacts remain local under `experiments/` and in W&B. The committed docs should explain what mattered and why.
