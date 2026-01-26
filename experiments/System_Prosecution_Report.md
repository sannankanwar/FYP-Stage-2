# Verdict: The People vs. FNO-ResNet18 Codebase

**Prosecutor:** Antigravity (Cortex AI)
**Date:** 2026-01-26
**Defendant:** `FYP-Stage-2` ML System

---

## Phase 0: Establish the Crime

The defendant makes the following implicit claims:
1.  **"I predict physical parameters."** (Implied by output layer architecture).
2.  **"Setting `standardize_outputs: true` helps training."** (Implied by config usage).
3.  **"My evaluation scripts tell the truth."** (Implied by their existence).

I will prove that claim #3 is false, and claim #2 actively sabotages claim #1 in downstream tools.

---

## Phase 1: Tensor Contract Ledger

| Tensor | Producer | Consumer | Expected Space | Status | Evidence |
|---|---|---|---|---|---|
| `model(x)` | `FNOResNet18` | `Trainer` | **Physical** | **PROVEN** | `HybridScaledOutput` forces physical range `[-500, 500]`. |
| `loss_input` | `Trainer` | `WeightedLoss` | **Physical** | **PROVEN** | Trainer passes native output to loss. |
| `loss_target` | `Trainer` | `WeightedLoss` | **Physical** | **PROVEN** | Trainer passes native target from loader. |
| `eval_pred` | `evaluate.py` | `plot_scatter` | **Physical** | **VIOLATED** | Script denormalizes physical tensor, creating **Exploded Space**. |
| `naive_loss` | `Trainer` | `NaiveLoss` | **Mixed** | **VIOLATED** | Compares `Pred(Physical)` vs `True(Normalized)`. |

**Finding:** The ledger contains two fundamental violations. One ensures silent failure (`NaiveLoss`), the other ensures noisy gaslighting (`evaluate.py`).

---

## Phase 2: Authority Interrogation

**Q: Who is allowed to scale outputs?**
**A:** The Definition says `HybridScaledOutput` (Model).
**A:** The Config says `standardize_outputs` (Trainer/Eval).

**Q: What happens if both do it?**
**A:** **Double Denormalization**.
The model scales latent -> physical.
The evaluator sees the flag and scales physical -> hyper-physical.
Example: $100 \mu m \times 500 (\sigma) = 50,000$.

**Conclusion:** Authority is split. The architecture ignores the config, but the tools respect the config. They are fighting.

---

## Phase 3: Training Is Not Evidence

The system currently trains successfully using `WeightedStandardizedLoss`.
*   **Why?** Because `WeightedStandardizedLoss` *internally* normalizes the physical prediction before comparing it.
*   **The Defense:** "But it works!"
*   **The Prosecution:** It works by accident of specific loss selection.
    *   If the user switches to `Naive5ParamMSELoss` (a valid config option), the model will receive gradients pushing it to output `0.2` instead of `100.0`.
    *   It will converge to near-zero.
    *   The user will see "low loss" (0.001) because $MSE(0.001, 0.2)$ is small compared to $MSE(100, 100)$.
    *   **Silent Failure.**

**Conclusion:** Training success is fragile and conditional.

---

## Phase 4: Evaluation Is Guilty

I execute `scripts/evaluate.py` with `expS10` config.
1.  `model(x)` returns `100.0`.
2.  Config has `standardize_outputs: true`.
3.  Script initializes `ParameterNormalizer`.
4.  Script runs `normalizer.denormalize_tensor(100.0)`.
5.  Result: `50,100.0`.
6.  Plot shows massive error bars.

**Conclusion:** `evaluate.py` is actively lying. It takes a correct model and reports it as broken. `evaluate.py` is **GUILTY**.

---

## Phase 5: Visualization Is a Weapon

The `Trainer` previously committed the same crime (Double Denormalization).
A patch was applied (defense exhibit A: Step 950).
However, the patch serves as admission of guilt. The underlying structural flaw (Flag vs Architecture mismatch) remains.

The visualization code now relies on a comment: `# Output is already physical`.
There is no assertion.
There is no metadata check.
If `FNOResNet18` is swapped for `StandardResNet` (no Hybrid layer), the plots will be wrong in the *other* direction (predicting 0.2 vs 100, plotted as 0.2 vs 100).

**Conclusion:** Visualization is structurally unsafe.

---

## Phase 6: Legacy Code Necropsy

**The Body:** `Naive5ParamMSELoss`
*   **Status:** Alive and importable.
*   **Trigger:** `loss_function: "naive_5param"` in config.
*   **Pathology:** Assumes inputs are comparable.
    *   Trainer handles `else` block: `current_target = normalizer(target)`.
    *   Model outputs Physical.
    *   Comparison: `Physical` vs `Normalized`.
*   **Verdict:** Code must be deleted. It is a trap.

---

## Phase 7: Counterfactual Attacks

**Attack 1:** Change `standardize_outputs` to `false`.
*   Result: `evaluate.py` works (accidentally). Model trains (if loss supports it).
*   But robust losses (Huber etc) might behave differently on raw scales (gradients depend on magnitude).

**Attack 2:** Load S10 checkpoint into `evaluate.py` but *forget* the config.
*   `evaluate.py` loads config from checkpoint.
*   If we manually override resolution or flags, we break the fragile matching.

**Attack 3:** Use `ResNet18` (non-FNO) which lacks `HybridScaledOutput`.
*   Model outputs unscaled logits? Or `tanh`?
*   `InverseMetalensModel` (legacy) assumes ...? (Unknown).
*   If it outputs unscaled, then `Trainer` *should* denormalize?
*   **Ambiguity:** The Trainer assumes *specific* architecture behavior (`FNOResNet18`) without checking it.

---

## Phase 8: Proof of Trustworthiness

Can I prove this system trusts?

1.  **Single Authority:** **NO**. Factory and Config disjoint.
2.  **No Conditional Scaling:** **NO**. `evaluate.py` has conditional logic.
3.  **Violations Crash:** **NO**. They produce bad math.

---

# Verdict

**The system cannot be trusted.**

While the current training loop (S10-S14) effectively optimizes the correct objective, the ecosystem around it is poisoned.
`evaluate.py` guarantees false rejection of valid models.
`Naive5ParamMSELoss` guarantees silent failure of training.
The code relies on implicit knowledge ("All models are HybridScaled") that is not enforced by the `Trainer` or `Config` schema.

**Sentence:**
1.  **Fix:** `scripts/evaluate.py` MUST be patched immediately to remove denormalization.
2.  **Delete:** `Naive5ParamMSELoss`.
3.  **Refactor:** Introduce `model.output_is_physical = True` property and assert it in Trainer/Eval.
