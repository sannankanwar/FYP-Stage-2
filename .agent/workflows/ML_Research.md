---
description: Research an ML method by finding papers (not blogs), triaging them for a 30GB VRAM budget, and producing a detailed experiment-ready report.
---

1. Capture the research target (concept + success criteria):
   - Write a 1-sentence definition of the concept/method.
   - Specify what “better results” means as: task + metric + dataset (or closest public benchmark).
   - State constraints: <= 30GB VRAM, single GPU unless noted, time budget.
   - Output: `target_spec` (definition, task, metric, dataset, constraints).

2. Build the concept card (quick understanding pass):
   - Explain the core mechanism in 2–5 bullets. (Mechanism = what it does differently internally.)
   - List 3–6 baseline methods it competes with.
   - List typical evaluation metrics and common benchmarks.
   - List 3–6 failure modes or “gotchas”.
   - Output: `concept_card`.

3. Generate keyword packs (search scaffolding):
   - Core keywords: canonical name, abbreviations, synonyms.
   - Method keywords: architecture/training objective/algorithm family terms.
   - Evaluation keywords: datasets, benchmarks, metrics.
   - Keyword mutations: generate 8–12 mixed variants.
   - Output: `keyword_packs` + `mutations`.

4. Create paper-only Google queries (10–14 queries):
   Use these constraints in each query set:
   - Prefer: `site:arxiv.org OR site:openreview.net OR site:aclanthology.org OR site:proceedings.neurips.cc OR site:proceedings.mlr.press`
   - Add: `filetype:pdf`
   - Exclude: `-blog -medium -substack -newsletter -towardsdatascience`
   - Time window: add year terms `2021 OR 2022 OR 2023 OR 2024 OR 2025 OR 2026` where helpful.
   Include at least:
   - `"<concept>" survey`
   - `"<concept>" benchmark`
   - `"<concept>" ablation`
   Output: `search_queries` (as exact copy-paste queries).

5. Harvest candidate papers (10–25 max):
   - Run the queries, collect results, and deduplicate by title.
   - Prefer: top venues (NeurIPS/ICML/ICLR/ACL/EMNLP/CVPR) + influential arXiv/OpenReview.
   - For each paper, capture:
     - Title, authors, year, venue/source, link
     - 1–2 sentence claim
     - What was evaluated (dataset/benchmark + metric)
     - Code availability (yes/no/unknown + link if known)
   - Output: `candidate_papers[]`.

6. Triage each paper with a rubric (score 0–2 each, total /10):
   For each paper assign:
   - Reproducible: code or strong implementation detail
   - Fit: improves the task/metric we care about
   - Compute-feasible: testable under 30GB VRAM (or with PEFT/quantization/smaller model)
   - Clarity: method is unambiguous enough to implement
   - Risk: unlikely to be fragile / over-tuned
   - Output: `triage_table` with score + 1–2 sentence justification.

7. Select top experiment picks (top 3):
   For each selected paper, write an experiment plan:
   - Hypothesis (what you expect to improve)
   - Minimal setup (smallest model + smallest dataset slice that still tests the claim)
   - Baseline (what you compare to)
   - Change (what you implement from the paper)
   - Success metric (exact number you will track)
   - VRAM strategy (examples: smaller model, LoRA/QLoRA, 8-bit/4-bit weights, shorter context, gradient checkpointing)
   - Expected failure modes
   - Output: `experiment_plans[3]`.

8. Produce the final detailed report:
   Must be in this structure:
   - ### 1) Concept card
   - ### 2) Keyword packs
   - ### 3) Paper-only search queries
   - ### 4) Candidate paper table (10–25)
   - ### 5) Top 3 experiment plans (30GB VRAM aware)
   - ### 6) Uncertainties + what would resolve them

// turbo
9. (Optional) Pull arXiv candidates quickly via script:
   `python scripts/arxiv_search.py "<your query>" 20`

Key Features
- Description: The agent uses this to match the workflow when the user asks for ML research that finds papers, filters by experiment feasibility, and outputs an experiment-ready report.
- // turbo: Put this annotation above a step to allow the agent to run it automatically without asking for confirmation (use carefully).
- // turbo-all: Put this anywhere in the file to auto-run all commands in the workflow.
