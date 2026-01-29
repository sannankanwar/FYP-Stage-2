# Old Experiments Archive

**Created:** 2026-01-28
**Status:** Frozen / Read-Only

This directory contains the history of Experiments Phases 1 through 5.
These files are kept for **provenance** and **reproducibility** of past results, but are **not** considered part of the active, maintained codebase.

## Structure
- `configs/`: YAML definitions for past runs.
- `scripts/`: Shell scripts and python helpers used to queue/run those specific experiments.
- `reports/`: Markdown notes, incident reports, and analysis results from those phases.
- `tests/`: Validation scripts that safeguarded specific experimental invariants (e.g. Phase 5 physics constraints).

## Usage
**Do not add new experiments here.**
If you need to reproduce a past result:
1. Copy the relevant config from `configs/phaseX/` to the root or a temporary location.
2. Verify if the paths in the config (e.g. `data_dir`) need updating.
3. Run using the modern `python src/main.py` entrypoint.
