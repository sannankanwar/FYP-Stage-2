# Experiment Archival and Repository Cleanup Plan

## 1. Rationale

The repository currently currently mixes "active" infrastructure with a large volume of "historical" experiment definitions. This creates several risks:
- **Cognitive Load**: Developers cannot distinguish between the "current best approach" and a failed experiment from 3 months ago.
- **Config Drift**: Legacy configs may break as the codebase evolves, creating a maintenance burden if they are treated as "live".
- **Namespace Pollution**: `scripts/` is cluttered with run-once shell scripts and specific analysis plotting tools.

By moving these artifacts to `Old_Experiments/`, we:
1.  **Freeze** their state (signaling "do not expect this to run out-of-the-box").
2.  **Preserve** knowledge (configs and results are kept for provenance).
3.  **Clean** the root for the new, standardized Phase 6+ experiments.

## 2. Target Folder Structure

The new `Old_Experiments/` directory will mimic the root structure to maintain logical grouping.

```text
Old_Experiments/
├── configs/
│   ├── phase1/                 # Was configs/experiments/
│   ├── phase2/                 # Was configs/experiments_2/
│   ├── phase3/                 # Was configs/experiments_3/
│   ├── phase4/                 # Was configs/experiments_4/
│   ├── phase5/                 # Was configs/experiments_5/
│   └── phase5_loss_study/      # Was configs/experiments_5_loss_study/
├── reports/                    # Historical markdown reports
├── scripts/                    # Run-once execution and plotting scripts
├── tests/                      # Experiment-verification tests
└── README.md                   # Explanation of this archive
```

## 3. Mapping Table

The following moves are planned. **No files are deleted.**

| Original Path | New Target Path | Reason |
| :--- | :--- | :--- |
| `configs/experiments/` | `Old_Experiments/configs/phase1/` | Legacy Phase 1 Baseline configs |
| `configs/experiments_2/` | `Old_Experiments/configs/phase2/` | Legacy Phase 2 Configs |
| `configs/experiments_3/` | `Old_Experiments/configs/phase3/` | Legacy Phase 3 Configs |
| `configs/experiments_4/` | `Old_Experiments/configs/phase4/` | Legacy Phase 4 Configs |
| `configs/experiments_5/` | `Old_Experiments/configs/phase5/` | Legacy Phase 5 Configs |
| `configs/experiments_5_loss_study/` | `Old_Experiments/configs/phase5_loss_study/` | Legacy Phase 5 Configs |
| `experiments/*.md` | `Old_Experiments/reports/` | Miscellaneous experiment notes |
| `reports/*.md` | `Old_Experiments/reports/` | Formal experiment reports |
| `run_experiments_5.sh` | `Old_Experiments/scripts/run_experiments_5.sh` | Phase 5 specific runner |
| `run_exp5_queue.sh` | `Old_Experiments/scripts/run_exp5_queue.sh` | Phase 5 specific runner |
| `scripts/run_experiments_nohup.sh` | `Old_Experiments/scripts/run_experiments_nohup.sh` | Generic legacy runner |
| `scripts/queue_experiments*.sh` | `Old_Experiments/scripts/` | Legacy queue scripts |
| `scripts/run_S09.sh` | `Old_Experiments/scripts/run_S09.sh` | Specific single-run script |
| `scripts/run_pure_regression_suite.sh` | `Old_Experiments/scripts/run_pure_regression_suite.sh` | One-off suite runner |
| `scripts/rank_models.py` | `Old_Experiments/scripts/rank_models.py` | Analysis script for legacy results |
| `scripts/compare_experiments.py` | `Old_Experiments/scripts/compare_experiments.py` | Analysis script for legacy results |
| `scripts/select_best_run.py` | `Old_Experiments/scripts/select_best_run.py` | Helper for legacy workflows |
| `cleanup_experiments.sh` | `Old_Experiments/scripts/cleanup_experiments.sh` | Legacy maintenance script |
| `tests/verify_exp5_invariants.py` | `Old_Experiments/tests/verify_exp5_invariants.py` | Experiment-specific validation |
| `tests/test_loss_variants.py` | `Old_Experiments/tests/test_loss_variants.py` | Experiment-specific validation |
| `tests/test_refinements.py` | `Old_Experiments/tests/test_refinements.py` | Experiment-specific validation |
| `tests/verify_losses.py` | `Old_Experiments/tests/verify_losses.py` | Experiment-specific validation |

**Retained in Root:**
- `src/*` (Core logic)
- `tests/test_integrity.py` (Core test)
- `tests/smoke_test_v2.py` (Core test)
- `tests/test_5param.py` (Core Core functionality verification)
- `tests/test_spectral_model.py` (Core model verification)
- `tests/verify_arch.py` (Core model verification)
- `configs/*.yaml` (Base configs: `data.yaml`, `model.yaml`, etc.)
- `scripts/train.py` (Main entrypoint)
- `scripts/evaluate.py` (Main entrypoint)
- `scripts/visualize_reconstruction.py` (Main entrypoint - useful general tool)

## 4. Classification Rules

1.  **Dependency Direction**: If `src/` code imports it, it stays. If it imports `src/` code but nothing depends *on* it (leaf node), it is archival candidate.
2.  **Naming Convention**: Filenames with "exp5", "experiments", "S09" (specific run IDs) are Archival.
3.  **Documentation vs. Code**: Markdown files in the root or `experiments/` folder that describe specific phases are Archival.

## 5. README for Old_Experiments

```markdown
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
```

## 6. Safety and Verification Checklist

- [ ] **Import Paths**: Check archived Python scripts (e.g., `Old_Experiments/scripts/rank_models.py`) for imports. Since they are moved deeper, `sys.path.append("..")` or relative imports might break.
    - *Mitigation*: We will not fix them immediately. They are archives. If needed, we recommend running them from root via `python Old_Experiments/scripts/script.py` and ensuring the script sets up path correctly, or leaving them broken until verified.
- [ ] **CI/CD**: Ensure no CI pipelines rely on `scripts/run_experiments_nohup.sh` existing in the exact location.
- [ ] **Config Relative Paths**: If configs use relative paths `../data`, moving them to `Old_Experiments/configs/phase1/` adds a directory level.
    - *Note*: Most configs likely use absolute paths or paths relative to the *execution* directory (CWD), so this is safe as long as we run from root.
```
