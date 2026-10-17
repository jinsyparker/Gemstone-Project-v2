# Gemstone dataset datasheet

Seed **42**. Split files: `train.csv`, `validation.csv`, `test.csv` (paths relative to the project root).
Full audit numbers: `audit_report.md`, `split_config.json`. Distribution table: `distribution_table.md`.

## Provenance
- Source: https://www.gempundit.com — loose-stone product photographs (studio backdrops).
- Scraped 2026-10-02 with `code/scrape.py` (User-Agent names the project and a contact address; 2 s between page
  requests, 0.75 s between image requests; every stage cached and resumable).
- Permission: the site owner's permission to scrape was reported by the project lead.
  **TODO (project lead): store the written confirmation here and state the agreed request rate.**
  robots.txt requests `Crawl-delay: 90`; we used the faster rate only on the strength of that permission.
- Server-side 403 rules for our bot User-Agent (not circumvented): paginated category pages (`/gemstones/<cat>/page/N`)
  and all `pigeon-blood-ruby-*` product URLs. Candidates therefore come from the public XML sitemap, and the
  *pigeon-blood ruby* sub-grade (127 of ~3,160 ruby listings, ~4%) is absent from the Ruby class.

## Classes (11)
Original 8: Peridot, Ametrine, Blue Zircon, White Sapphire, Yellow Sapphire, Ruby, Emerald, Blue Sapphire.
Added 3 for inventory depth and confusability with an existing class: Pink Sapphire (↔ Ruby), Amethyst (↔ Ametrine),
Citrine (↔ Yellow Sapphire).

Label = product-URL stem (mapping in `config.yaml`) **and** product-page title must name the class. Excluded: star
ruby/sapphire (asterism cabochons), pairs/sets/lots, jewellery. Stem→class agreed with the site's own category pages on
524/528 sampled items (the 4 misses were SKUs newer than the sitemap; no conflicts).

## What was kept (5,526 images, 1,790 stone groups)
Only photographs named `gp<id>-<n>-…` (numbered stone views). Excluded by filename role: `-hand-` (hand/ring shots,
1,523) and `-…-certified-` (lab certificates, 1,732: they contain printed text, a label shortcut).
Up to ~500 images per class were collected (random SKU order, seed 42), so class sizes are 497–521: **balanced by design**
(imbalance ratio ≈ 1.05), not the natural inventory distribution.

## Cleaning (8 images removed of 5,534 downloaded)
- Corrupt / truncated / tiny / odd-aspect: **0**.
- Promo mock-up banner ("Your gemstone, after being crafted into a ring…", a stock image shown under Amethyst and
  Blue Sapphire): **5**, removed by similarity to curated exemplars (`audit.template_exemplars`).
- Exact duplicate inside one SKU: **3**.
- Cross-class duplicates: **0**. Components flagged giant: **0**.

## Splits (70/15/15, stratified by class, **grouped by physical stone**)
- A group = one SKU (all photos of one stone) plus any other SKUs linked as same/sister stone or same photo.
- Cross-SKU links: 44 edges, 31 multi-SKU groups, none across classes. Rule (thresholds chosen on contact sheets):
  SIFT+RANSAC inliers ≥ 20 **and** stone-colour distance ≤ 0.06, **or** DINOv2 cosine ≥ 0.97 **and** stone colour ≤ 0.05.
  Lessons recorded in `config.yaml`: embedding cosine alone is >0.95 for any similar-cut gems on the grey backdrop;
  SIFT alone matched identical overlay text ("Yellow Light") and same-cut/different-colour stones.
- Checks (all pass): no SKU / group / file hash in more than one split; per-class ratio drift ≤ 0.2 pts; re-running the
  split is byte-identical; a different seed gives a different split.
- **Effective test size:** 826 test images but only **23–31 independent stones per class** (≈300 total). Images of one
  stone are correlated, so confidence intervals must be bootstrapped over **stones (group_id)**, not images.
- `test.csv` and the test images are read-only; `split_config.json` stores the CSV SHA-256 values.

## Known biases / caveats (important for the paper)
- **Background/style shortcut.** Border colour, image size and brightness alone (no stone pixels; random forest trained
  on train, scored on validation) reach **39.6 % accuracy / 0.397 macro-F1** versus 9.1 % chance. White Sapphire is on a
  black backdrop in 97 % of its photos (grey for almost every other class) and is recovered with 95 % recall from
  background alone; 45 % of Blue Sapphire photos use a white backdrop. Report this, and consider background
  normalisation / background-randomising augmentation applied identically to all models.
- Different listing styles carry overlays/props (e.g. "Yellow Light"/"Back Light" captions, tweezers photos, mostly Ruby).
- Sister stones from one parcel (consecutive SKUs) are near-identical; those found are grouped together.
- Product-photo domain only; labels are the vendor's, not independently gemmologist-verified.

## Note on earlier numbers
The earlier manuscript quotes 3,730 images / a class table summing to 3,182 across 8 classes. The original files were
not retained; this is a fresh scrape of a changing inventory with three additional classes, so counts differ. The new
counts supersede them.

## Benchmark classes (Deliverable 2 onward): 10, without White Sapphire
White Sapphire is excluded from model benchmarking. 97 % of its photographs have a black backdrop while nearly every
other class is on grey, so backdrop colour alone recovers it with 95 % recall. The benchmark therefore uses 10 classes:
**3,518 train / 755 validation / 751 test** images, each stone keeping the split it has in the files above.
Derived lists: `dataset/benchmark/{train,validation,test}.csv`, `classes.json`, `benchmark_split.json` (SHA-256s).
The 11-class files above are unchanged. Backdrop colour, image size and brightness alone still give 35 % validation
accuracy on the 10 classes (chance 10 %), mostly through Blue Sapphire, so a background-only control row is reported
next to the benchmark.

## Background dependence (Deliverable 2b)
Tested on the 10-class benchmark with five frozen baselines and a small retrain arm (details in `NOTES.md`, results in
`results/background_dependence.csv`). Classification relies mainly on the gemstone: removing the backdrop costs 0.0 to 2.9
macro-F1 points, and backdrop-only images are classified at chance by the frozen models. The backdrop is nevertheless
informative (a MobileNetV2 trained on backdrop-only images scores 32.8% against 10% chance), and the models carry a
learned link between a white backdrop and Blue Sapphire (51 to 71% of predictions that change when a stone is moved onto
white become Blue Sapphire). Masks come from GrabCut and may include a thin sliver of the stone's reflection at its base.
