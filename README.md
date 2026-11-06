# Gemstone Extension

Controlled benchmark, a dual-branch colour/texture model, and a robustness and background-reliance study for gemstone image
classification. The study is framed around **robustness and background reliance at matched accuracy**: the question is not only how
accurate the models are, but whether they look at the stone, and how much they lose when the photograph is disturbed.

Ten classes (Peridot, Ametrine, Blue Zircon, Yellow Sapphire, Ruby, Emerald, Blue Sapphire, Pink Sapphire, Amethyst, Citrine),
5,024 loose-stone photographs of 1,629 physical stones, split 70/15/15 **by stone**.

## What is here
| Part | Where |
|---|---|
| Dataset pipeline, audit, grouped split, datasheet | `code/scrape.py`, `code/preprocessing.py`, `dataset/` (`DATASHEET.md`, `data_trace.md`, `white_sapphire_exclusion.md`) |
| Controlled five-model benchmark (ResNet50, DenseNet121, MobileNetV2, EfficientNetV2-S, ConvNeXt-Tiny) | `code/baselines.py`, `code/train.py`, `code/evaluate.py`, `results/benchmark.csv` |
| Background-dependence experiment (stone only, backdrop only, backdrop swaps) | `code/background_analysis.py`, `results/background_dependence*.csv` |
| Proposed model: EfficientNetV2-S plus a colour/texture branch with gated fusion | `code/proposed_model.py`, `figures/architecture.png` |
| Robustness suite (18 perturbations) | `code/robustness.py`, `results/robustness*.csv` |
| Seeds, ablation (A to E and a fully fine-tuned control) | `code/runs.py`, `code/run_queue.sh`, `results/PREREGISTRATION.md` |
| One test pass per checkpoint, tables, Grad-CAM, error analysis, efficiency | `code/final_eval.py`, `code/final_tables.py`, `code/seedstats.py`, `code/gradcam.py`, `code/error_analysis.py`, `code/efficiency.py` |
| Earlier studies and what changed | `docs/earlier_studies.md` |
| Decisions and findings, deliverable by deliverable | `NOTES.md` |

## Data policy
The photographs belong to the vendor (gempundit.com) and are **not redistributed**. The repository holds the file lists
(`dataset/{train,validation,test}.csv`, the 10-class lists in `dataset/benchmark/`), their SHA-256 values, the source URL of
every image (`dataset/scraped_images.csv`) and the code that rebuilds the dataset. The `dataset/raw/`, `dataset/_cache/`
folders and the trained checkpoints (`models/**/*.pt`) are git-ignored. Labels are the vendor's product categories, not
independently verified by a gemmologist.

## Setup
Python 3.11. Tested on an Apple M1 (PyTorch MPS); CPU works, slowly (`GEM_DEVICE=cpu`).

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests -q          # about 90 tests, CPU only, no images needed except where a test says so
```

## Reproducing the pipeline
```bash
python code/scrape.py all                    # discover products from the sitemap, download stone photographs (resumable, throttled)
python code/preprocessing.py all             # QC, duplicates, stone groups, grouped split, tables, audit report
python code/preprocessing.py benchmark       # the 10-class lists (White Sapphire excluded)
python code/preprocessing.py cache           # 256 px image cache used by every model
python code/data_trace.py                    # image trace and the White Sapphire backdrop evidence
bash code/run_baselines.sh                   # five baselines, one frozen recipe (config.yaml)
python code/background_analysis.py masks     # stone masks, then: variants (backdrop swaps), final, figure
python code/train.py --model Proposed        # the proposed model
bash code/run_queue.sh                       # seeds 43 and 44 and the ablation, about 10 h on an M1, resumable
```

## Test-set discipline
- The test split is never loaded by training code; evaluation code needs an explicit flag.
- Every test read is appended to `results/test_access_log.json`.
- `results/PREREGISTRATION.md` fixes the selection rules, statistics and comparisons **before** any new run or test read; its
  hash is in `results/FREEZE_prereg.json` and `tests/test_freeze.py` fails if it changes.
- Checkpoints are frozen with SHA-256 manifests (`results/FREEZE*.json`, `python code/freeze.py check`).
- `code/final_eval.py` reads the test set once per checkpoint and only after every run is trained and frozen:
  ```bash
  python code/freeze.py make runs
  python code/final_eval.py --all --split test --test
  python code/final_tables.py --split test
  python code/error_analysis.py --split test
  python code/efficiency.py                    # machine idle and plugged in; protocol in docs/efficiency_protocol.md
  ```
  Tables are rebuilt from the saved outputs in `results/final/`; the images are not read again.

## Statistics
Confidence intervals resample **stones**, not images, because about three photographs show the same stone (the test set has about
290 independent stones). Differences between models are paired: every model and seed is scored on the same resampled stones. A claim
that two models differ needs the 95% interval of the seed-mean difference to exclude zero **and** the same sign in every seed.
Grad-CAM is reported as supporting evidence only.

## Results
Test set, three seeds (42 to 44), intervals resample stones; a difference is claimed only if the 95% interval excludes zero and every
seed agrees (`results/PREREGISTRATION.md`). Tables: `results/final_tables/`, `results/error_analysis/`; figures: `figures/`.

| | Proposed | ConvNeXt-Tiny | EfficientNetV2-S |
|---|---|---|---|
| Clean macro-F1 | 96.5% | 96.6% | 95.1% |
| Core perturbations within 2 points (of 8) | 8 | 8 | 6 |
| Stone only, change in macro-F1 | -1.0 | 0.0 | -2.9 |
| Stone on a white backdrop, change | **-12.6** | -3.2 | -6.6 |
| Changed answers that become Blue Sapphire (white) | 73% | 49% | 51% |

- Accuracy is matched with ConvNeXt-Tiny (no clear difference) and higher than the proposed model's own RGB backbone (+1.4 points).
- Robustness to photographic perturbations matches ConvNeXt-Tiny; the proposed model is more robust than EfficientNetV2-S under noise, blur and brighter light.
- The proposed model is **more sensitive than both baselines to a white backdrop**, although Grad-CAM puts more of its attention on the stone.
- Ablation: the colour/texture branch adds +1.6 points over a fully fine-tuned RGB-only model at equal training; gating is no better than concatenation.
- Efficiency (`results/efficiency.csv`): 21.8 M parameters, 6.2 GFLOPs, 29.6 ms per image on the M1 GPU at batch 1.

The earlier single-seed results are in `results/benchmark.csv`, `results/robustness*.csv` and `NOTES.md`.

## Limitations
One vendor's studio photographs; product-page labels; ten classes; three seeds. The trained checkpoints are not in the repository.

## Licence
MIT (see `LICENSE`). The licence covers the code and the derived tables, not the photographs.
