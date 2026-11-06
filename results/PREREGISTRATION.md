# Pre-registration of the final analyses

Written 2026-11-05, before any of the new training runs and before any new test-set read. Its SHA-256 is stored in
`results/FREEZE_prereg.json`; `tests/test_freeze.py` fails if this file changes afterwards. Anything done differently
from this document is listed in the final report as a deviation.

## 1. What is run
- **Main comparison:** Proposed (EfficientNetV2-S + colour/texture branch, gated fusion), ConvNeXt-Tiny and
  EfficientNetV2-S, seeds **42, 43 and 44**. Seed 42 already exists from Deliverables 2 to 4; seeds 43 and 44 are new.
- **Ablation**, seeds 42, 43, 44, using the Proposed code with flag overrides (`code/runs.py`):
  A RGB branch only, backbone frozen; B colour/texture branch only; C both branches concatenated; D both branches with
  gated fusion (A to D: stage 1 only, 10 epochs); **E = the main Proposed model** (gated fusion, two-stage recipe);
  **A-FT = RGB branch only with the full two-stage recipe** (the fully fine-tuned control).
- Recipe: exactly `config.yaml` (`benchmark:` and `proposed:`), unchanged. No hyper-parameter is tuned for any run.
  The 10-class split, test list and checkpoint rule are those of Deliverable 2.

## 2. Selection rules (nothing below may depend on a test-set number)
- A run's checkpoint is the epoch with the best **validation** macro-F1. No other selection step exists.
- A crashed run is resumed with the same seed. A run that trains badly is reported as it is; it is never re-seeded,
  re-run with different settings, or dropped.
- All checkpoints are trained and frozen (SHA-256 manifest) **before** the first new test read.
- Each checkpoint then gets **one logged test pass** (`code/final_eval.py`) that produces every test output below.
  Re-running the analysis rebuilds the tables from the saved outputs and does not read the test images again.

## 3. Test outputs per checkpoint
1. Clean logits (accuracy, macro-F1, per-class metrics, confusion matrix).
2. The 18 perturbation conditions of Deliverable 6 (brightness, contrast, noise, rotation, flip, blur) with the same
   definitions as `code/robustness.py`; the 8 core conditions are the headline set.
3. The background variants of Deliverable 2b: original, stone only, backdrop only, backdrop swapped to white, black and
   another class's backdrop.
4. Grad-CAM attention measured against the stone masks: stone attention share, area-normalised enrichment, top-20% pointing.

## 4. Statistics
- **Unit of resampling is the stone** (`group_id`), never the image. Each bootstrap draw (1,000 draws, seed 42) resamples
  stones with replacement and recomputes every statistic on the same stones for every model and seed, so differences are paired.
- **Seed-mean statistic:** for each model and condition, the mean over the three seeds of the metric. Its interval comes from
  the stone bootstrap above (test-set sampling). The **SD across seeds** (n = 3, a rough guide only) is reported beside it,
  together with each seed's own value.
- McNemar's test is not used: images of one stone are not independent, so the stone bootstrap is the test.
- **A claim that two models differ** needs both (a) the 95% paired interval of the seed-mean difference to exclude zero and
  (b) the difference to have the same sign in every individual seed. Anything else is reported as "no clear difference".
- **Robustness rule (unchanged from Deliverable 6):** a model is robust to a condition if its macro-F1 drops by 2 points or
  less from its own clean result. "More robust / more fragile" compares the drop of two models (paired interval of the
  difference in drops, same two-part rule).

## 5. Primary comparisons (fixed; everything else is descriptive)
Proposed against ConvNeXt-Tiny and against EfficientNetV2-S, on:
1. clean test macro-F1 (also accuracy);
2. the 8 core robustness conditions;
3. background: macro-F1 change on stone-only images, macro-F1 on backdrop-only images, change when the stone is placed on a white backdrop.

Ablation comparisons, in this order: **E vs A-FT** (the colour/texture branch and gates at equal training budget; primary),
E vs D (effect of fine-tuning), D vs C (gating against concatenation), D vs A (adding the colour/texture branch with the
backbone frozen). Each is reported on clean macro-F1 and the 8 core robustness conditions.

With 16 robustness comparisons in item 2 alone, intervals are reported unadjusted and the two-part rule in section 4 is
the guard against chance findings; no single interval is presented as a headline.

## 6. Grad-CAM and error analysis
- Grad-CAM is supporting evidence. Layers: ResNet50 `layer4`, DenseNet121 final features, MobileNetV2 and EfficientNetV2-S
  last convolution, ConvNeXt-Tiny last stage, Proposed fused map before pooling. It is not used to decide any comparison.
- A parameter-randomisation sanity check is run on one model; if the maps do not change, Grad-CAM is reported as uninformative.
- Error analysis (confusion matrices, per-class metrics, top confused pairs, examples) is written on validation outputs and then
  run once on the test outputs. Error examples are a fixed-seed random draw per confused pair, not chosen by eye.

## 7. Efficiency
Parameters, FLOPs and latency are measured once, at the end, under the protocol written in `docs/efficiency_protocol.md`
(M1 GPU and CPU, fp32, 224 px, synthetic input, batch 1 and 32, 20 warm-up and 200 timed iterations, machine idle and plugged in).

## 8. What was already seen (honest labelling)
- Seed-42 clean test results and the Deliverable 6 robustness results of the five baselines and the Proposed model were seen
  before this document. The Proposed architecture was fixed on **validation** numbers only (Deliverables 3 and 4). Before
  Deliverable 6, the plan expected the Proposed model to be less robust than the baselines under noise, blur, brightness and
  contrast; the seed-42 result did not support that (`NOTES.md`, Deliverable 6).
- Seeds 43 and 44 are therefore the confirmatory set. Results are reported for all three seeds and, separately, for seeds 43
  and 44 only.
- The Deliverable 2b backdrop tests of the five baselines were run on the test set before this document; the Proposed model's
  background tests on the test set are new.
