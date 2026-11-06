# Earlier gemstone classification studies (our own)

Scope: our own earlier work only. Each entry lists what was verified in the source itself. The point of this note is to be
exact about where the earlier evaluations lacked an independent test set, because that changes how their numbers can be compared
with this project's.

## 1. Conference paper: "Enhancing Classification of Gemstones through Deep Learning Analysis: A Comparative Study"
Patel, Banker and Rao (Manipal Institute of Technology). Source: the revised manuscript (`GemConfPaper_revised`).

| | |
|---|---|
| Data | 3,182 images in **8 classes** (Table 1: Peridot 480, Ametrine 262, Blue Zircon 246, White Sapphire 466, Yellow Sapphire 442, Ruby 444, Emerald 446, Blue Sapphire 396), from gempundit.com product photographs |
| Split | **2,545 training and 637 validation images** (a single 80/20 division; no stone- or product-level grouping is described) |
| Models | ImageNet-pretrained networks with custom classification layers: ResNet50, EfficientNet (named "EfficientNet50"), MobileNetV2, VGG16 (InceptionV3 is also named in the abstract) |
| Training | Adam, categorical cross-entropy; early stopping on validation loss; epoch count stated as five in one place and ten in another |
| Reported scores (Table 2) | accuracy 97.58 (ResNet50), 97.02 (EfficientNet), 95.29 (MobileNetV2), 97.65 (VGG16) |
| Independent test set | **None.** The 637 held-out images served as validation data during training (early stopping) and as the evaluation set for every reported score; the "test accuracy" curves in the figures are those same images. |
| Other inconsistencies | the text says "11 classes" while naming seven; images are described as 156 x 250 px and later as 224 x 224 px; the Table 2 precision, recall and F1 of ResNet50 are identical to MobileNetV2's and do not match ResNet50's own accuracy |

## 2. Public notebook: `jinsyparker/Gemstone-Classification` (March 2024)
Source: `Gemstone_Final.ipynb` in the repository ("Classify 8 types of Gemstones using Transfer Learning approach").

| | |
|---|---|
| Split | one call to `train_test_split(images, labels, test_size=0.2, random_state=42)` on individual images: no stratification and no grouping by stone |
| Models | transfer learning with DenseNet169, EfficientNetB0, MobileNet, ResNet and VGG backbones, 224 x 224 input, Adam |
| Training | 5 epochs, batch size 32, `EarlyStopping(monitor='val_loss', patience=3)` |
| Evaluation | `validation_data=(X_test, y_test)` during training, and the same `X_test` for the final predictions and classification report |
| Reported scores | validation accuracy per epoch between about 0.95 and 0.98 |
| Independent test set | **None.** The held-out 20% is both the early-stopping signal and the evaluation set. |

## What this implies
- **Optimistic selection.** Early stopping and reporting on the same images means the reported accuracy is a validation score
  chosen at its best epoch, not an estimate of performance on unseen data.
- **Possible leakage.** Both earlier splits were random over images. A product has several photographs of the same physical stone
  (Deliverable 1 found about 3 per stone), so near-identical images can sit on both sides of a split and inflate scores.
- **Not comparable.** The earlier scores (95 to 98%) use 8 classes, a different image set and an evaluation set that was also used
  for selection. They are shown in `results/benchmark.csv` as historical rows and are not compared with this project's test scores.

## What this project changes
| Earlier studies | This project |
|---|---|
| Random image-level 80/20 split | 70/15/15 split **grouped by physical stone** and stratified by class; near-duplicate stones linked and kept together; zero overlaps across splits |
| Evaluation set also used for early stopping | Early stopping and checkpoint choice on a **validation** split; the **test** split is read once per model and every access is logged |
| One run, point estimate | Three seeds per key model, mean and spread, intervals that resample stones |
| No check of what the model looks at | Background-dependence experiments, perturbation tests, and Grad-CAM attention scored against the stone mask |
| Models trained with different, loosely specified settings | One frozen recipe for every model, written down before the comparison |
| 8 classes including White Sapphire | 10 classes; White Sapphire excluded because 97% of its photographs have a black backdrop (see `dataset/white_sapphire_exclusion.md`) |
