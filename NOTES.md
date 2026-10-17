# Project notes

## Supervisor's execution order
Directive received after Deliverable 2. Status is updated as steps finish.

- [x] Deliverable 1: dataset pipeline, audit, frozen split (11 classes; seed 42; grouped by stone).
- [x] Deliverable 2: controlled baseline benchmark on 10 classes (ConvNeXt-Tiny best, 0.9672 test macro-F1, single seed).
- [x] **White Sapphire removed for now** (10-class problem). See below.
- [x] **Freeze the split and the five baseline models** (`results/FREEZE.json`, checked by `tests/test_freeze.py`).
- [x] **Deliverable 2b: background-dependence experiment** (done; see below).
- [x] Deliverable 3 and 4: proposed dual-branch Colour-Texture Attention network (`code/proposed_model.py`, `figures/architecture.png`) and its two-stage training (`models/Proposed/best.pt`, `results/baseline_history/Proposed.json`). **Validation only so far; the test set is untouched for this model.**
- [ ] Deliverable 5: ablation, variants A to E.
- [x] Deliverable 6: robustness experiments (`code/robustness.py`, `results/robustness*.csv`, `figures/robustness.png`; proposed model's first test read; frozen in `results/FREEZE_proposed.json`). See below.
- [ ] Deliverables 7 and 8: class-wise error analysis and Grad-CAM (with foreground/background attention).
- [ ] Deliverable 9: repeat the proposed model and the strongest 2 to 3 baselines with at least 3 seeds, mean ± SD, stone-level bootstrap intervals.
- [ ] Deliverable 10: final package and private GitHub repository.

## Decisions on record
- **10 classes from Deliverable 2 onward.** White Sapphire is excluded because 97% of its photographs have a black backdrop while the other classes are on grey; backdrop alone recovers it with 95% recall. The switch is `benchmark.exclude_classes` in `config.yaml`. It is *not* free to reverse: the cache must be rebuilt and every model retrained. The 11-class Deliverable 1 files are unchanged.
- **Test set rules.** Selection of recipes, epochs and thresholds uses validation only. Each reported test evaluation is logged in `results/test_access_log.json`. Test images are never inspected by eye; masks and image variants for them are produced blind.
- **Historical rows** (earlier paper, as reported) sit in a separate block of `results/benchmark.csv`. ResNet50's precision, recall and F1 match MobileNetV2's and do not match its own accuracy, and "EfficientNet50" is not a standard variant: check against the source.
- **Seeds.** The Deliverable 2 baselines are seed 42. Deliverable 9 adds seeds 43 and 44 (at least).
- **Choice of "strongest baselines" for Deliverable 9** is made by validation macro-F1, not test.

## Deliverable 2b: background-dependence results (test set, seed 42)
Code `code/background_analysis.py`; results `results/background_dependence*.csv`, `results/background_flips.csv`; figures `figures/background_dependence.png`, `figures/background_examples.png`.
- **Pre-stated rule** (limited reliance if stone-only macro-F1 is within 5 points of original *and* backdrop-only macro-F1 is below 35%): **met by all five frozen baselines.** Stone-only costs 0.0 to 2.9 points; backdrop-only scores 6 to 11% (chance 10%).
- **Stone carries nearly all the signal.** MobileNetV2 retrained on stone-only images: 94.8% on stone-only (95.9% for the original model).
- **Backdrop alone is learnable.** MobileNetV2 retrained on backdrop-only images: 32.8% (95% interval 28.6 to 36.2), about 3.3 times chance, in line with the pixel-statistics control (30.5%).
- **Sensitivity to unusual backdrops.** Stone on flat white or black costs 3 to 22 points. ConvNeXt-Tiny is most robust (-3.4 white, -5.2 black); MobileNetV2 (-20.0 white) and EfficientNetV2-S (-22.1 black) are the most sensitive.
- **Specific association.** On a white backdrop 51 to 71% of changed predictions go to Blue Sapphire for all five models (Blue Sapphire is the only class with many white-backdrop photos, 45%).
- **Open decision for Deliverable 3.** The proposed model's HSV branch sees the backdrop colour. Options: (i) raw images and report the finding, (ii) backdrop-neutralised images for all models (re-run baselines), (iii) colour branch fed foreground pixels only (masks needed at inference). Decide with the numbers in hand.
- **Method notes.** Masks: GrabCut seeded by a fitted backdrop model, cast-shadow rule, convex-hull repair; checked by eye on 100 validation images (test images never viewed). "Backdrop only" removes the stone *and* its shadow and coloured reflection (an earlier version leaked stone colour through the reflection). Test set read once per model and image version, logged in `results/test_access_log.json`.

## Deliverables 3 and 4: proposed model, training outcome (validation only, seed 42)
- Architecture: EfficientNetV2-S (Branch A, pretrained) + a 1 M-parameter CNN on a 6-channel hue/saturation/value + Sobel + Laplacian map (Branch B), 1x1 projections to 256 channels, gated attention fusion, global average pool, dropout 0.3, linear to 10 classes. 21.8 M parameters. Same recipe, split and seed as the baselines.
- Gates are initialised near zero so every gate starts at about 0.5 (a first version saturated at 0 or 1; caught by a test and fixed; checked with real weights on real validation images: gates 0.39 to 0.60 at step 0).
- Training: early-stopped at epoch 22, **best validation macro-F1 0.9737 at epoch 21** (ConvNeXt-Tiny 0.9723, DenseNet121 0.9666). A 0.14-point gap is about one validation image: not a real difference.
- Behaviour: already about 0.95 to 0.97 validation macro-F1 inside the head-only stage; fine-tuning adds little (stage 1 best 0.9698, epoch 9). Expect the D5 ablation to show a small gain from fine-tuning (variant E over D).
- **Backdrop tests on the proposed model (validation).** Stone-only -0.3 pts (most robust of all six models), stone on black -1.3 (second-best after ConvNeXt, whose -3.5 is the baselines' best), another class's backdrop -1.3; **stone on white -10.6** (second-worst after MobileNetV2 at -19.8). On white, 75.5% of the 102 changed predictions go to Blue Sapphire (baselines 51 to 71%), mostly blue zircons (41). So the colour branch reduces dependence on the backdrop in general but strengthens the white-backdrop to Blue Sapphire link. To be confirmed on test when the final model is evaluated. Results: `results/background_dependence_validation_proposed.csv`.
- Policy chosen with the user: raw images, with the stone-only / backdrop-only / swap tests repeated on every new model.

## Deliverable 6: robustness (test set, seed 42, one run per model)
- Setup: 18 conditions (brightness and contrast x0.6/0.8/1.2/1.4, noise sigma 5/10/20, rotation +-5/10/20 deg, horizontal flip, blur sigma 0.5/1/2) on the cached 256 px crops, then the unchanged evaluation transform. All six models. Rules fixed before any number was seen: robust = macro-F1 drop within 2 points; "more fragile" only if the paired stone-level 95% interval of (proposed drop - baseline drop) lies entirely below zero; formal comparisons only against ConvNeXt-Tiny and EfficientNetV2-S. Code was developed on validation images only; the test set was read once per model (7 new entries in `results/test_access_log.json`).
- **Clean test macro-F1 of the proposed model: 0.9612** (accuracy 0.9614, 95% CI 0.940 to 0.978). ConvNeXt-Tiny 0.9672. The proposed model is not better than the best baseline on test; the gap is inside the interval.
- **All 8 core conditions are within 2 points for the proposed model** (largest drop: blur sigma 1, -1.6). Same for ConvNeXt-Tiny. Outside 2 points: EfficientNetV2-S 3 conditions, MobileNetV2 2, ResNet50 2, DenseNet121 1 (core, brightness/contrast/noise/rotation/flip/blur at the brief's level).
- Against **EfficientNetV2-S** (its RGB-only counterpart): proposed more robust in 8 of 18 conditions (noise sigma 10: +10.4 pts, blur sigma 1: +4.1, brightness +20%: +3.0), more fragile in none. Against **ConvNeXt-Tiny**: no difference in any core condition; proposed more fragile only at brightness +40% (-2.2, a severity-sweep condition, one of 18 comparisons, not corrected for multiplicity).
- Stated expectation was wrong: I predicted the Sobel/Laplacian/HSV inputs would make the proposed model less robust to noise, blur, brightness and contrast than the baselines. It was not. Do not explain why until D5 shows what each branch contributes; one possibility is that the model leans on the RGB branch under corruption, another is that it is a property of this EfficientNetV2-S run (single seed). D9 seeds will show whether the EfficientNetV2-S noise fragility is specific to seed 42.
- Mild noise sigma 5 *raised* proposed macro-F1 by 1.1 (interval just above zero). Treat as noise in the estimate, not an effect.
- Caveats: single seed, about 290 independent stones, so one-point differences are not interpretable; perturbations are synthetic, not real camera or lighting variation; flips and 90-degree rotations were in training augmentation for every model.
- D5 (ablation) remains outstanding; the clean test predictions of the proposed model are saved at `results/test_predictions/Proposed.npz` and must not be re-read for ablation variants.
