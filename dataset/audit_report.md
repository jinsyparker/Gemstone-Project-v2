# Dataset audit report

- Seed: **42**   Embedding model: `vit_small_patch14_dinov2.lvd142m`
- Images kept: {'train': 3870, 'validation': 830, 'test': 826}  (groups: 1790; removed: 8)
- Removed by reason: {'non_stone_template': 5, 'exact_dup_same_sku': 3}
- Class imbalance ratio (max/min): **1.05**, normalised entropy: **1.000**
- Max split-ratio drift vs 70/15/15: 0.20 pts (tolerance 3.0)

## Leakage checks (all must be 0)

```
{
 "group_id_in_multiple_splits": 0,
 "file_sha256_in_multiple_splits": 0,
 "pixel_sha256_in_multiple_splits": 0,
 "sku_in_multiple_splits": 0,
 "near_dup_edges_crossing_splits": 0
}
```
Max cosine similarity val/test -> train: `{'validation': {'max': 0.9731, 'p99': 0.9688, 'median': 0.9322}, 'test': {'max': 0.9776, 'p99': 0.9686, 'median': 0.9311}, 'test_vs_validation': 0.9719}`

## Distribution

| Class           |   train |   validation |   test |   Total |   train_% |   validation_% |   test_% |   Groups |
|:----------------|--------:|-------------:|-------:|--------:|----------:|---------------:|---------:|---------:|
| Peridot         |     351 |           75 |     76 |     502 |      69.9 |           14.9 |     15.1 |      167 |
| Ametrine        |     351 |           75 |     75 |     501 |      70.1 |           15   |     15   |      171 |
| Blue Zircon     |     352 |           75 |     75 |     502 |      70.1 |           14.9 |     14.9 |      167 |
| White Sapphire  |     352 |           75 |     75 |     502 |      70.1 |           14.9 |     14.9 |      161 |
| Yellow Sapphire |     351 |           76 |     75 |     502 |      69.9 |           15.1 |     14.9 |      164 |
| Ruby            |     364 |           79 |     78 |     521 |      69.9 |           15.2 |     15   |      127 |
| Emerald         |     351 |           75 |     75 |     501 |      70.1 |           15   |     15   |      153 |
| Blue Sapphire   |     350 |           75 |     74 |     499 |      70.1 |           15   |     14.8 |      160 |
| Pink Sapphire   |     348 |           75 |     74 |     497 |      70   |           15.1 |     14.9 |      167 |
| Amethyst        |     348 |           75 |     74 |     497 |      70   |           15.1 |     14.9 |      172 |
| Citrine         |     352 |           75 |     75 |     502 |      70.1 |           14.9 |     14.9 |      181 |
| Total           |    3870 |          830 |    826 |    5526 |      70   |           15   |     14.9 |     1790 |

## Per-class image statistics (bias scan; kept images)

| class           |   n |   width |   height |   bytes_kb |   brightness |   bg_r |   bg_g |   bg_b |
|:----------------|----:|--------:|---------:|-----------:|-------------:|-------:|-------:|-------:|
| Amethyst        | 497 |    1920 |     1200 |         74 |        141.6 |  149.2 |  149.4 |  150.7 |
| Ametrine        | 501 |    1920 |     1200 |         60 |        136.4 |  134.3 |  134.2 |  136.8 |
| Blue Sapphire   | 499 |    1920 |     1200 |         74 |        183   |  201.9 |  202   |  202.6 |
| Blue Zircon     | 502 |    1920 |     1200 |         76 |        138.2 |  136.8 |  136.7 |  138.9 |
| Citrine         | 502 |    1920 |     1200 |         73 |        141.6 |  145.5 |  145.9 |  147.6 |
| Emerald         | 501 |    1920 |     1200 |         71 |        143.1 |  152.5 |  152.4 |  154.3 |
| Peridot         | 502 |    1920 |     1200 |         65 |        133.3 |  138.5 |  138.7 |  141.5 |
| Pink Sapphire   | 497 |    1920 |     1200 |         69 |        141.1 |  136.8 |  137.2 |  139.8 |
| Ruby            | 521 |    1920 |     1200 |         71 |        134.1 |  147   |  146.9 |  148.1 |
| White Sapphire  | 502 |    1920 |     1200 |         68 |         44.9 |    7.6 |    7.6 |    8.4 |
| Yellow Sapphire | 502 |    1920 |     1200 |         74 |        150   |  148   |  147.2 |  147.7 |

## Thresholds

```
{
 "cache": "dataset/_cache",
 "workers": 6,
 "min_side": 256,
 "max_aspect": 3.0,
 "embed_model": "vit_small_patch14_dinov2.lvd142m",
 "embed_fallbacks": [
  "resnet50.a1_in1k"
 ],
 "embed_size": 224,
 "embed_batch": 32,
 "pair_floor_cos": 0.92,
 "pair_floor_hamming": 8,
 "sift": {
  "max_side": 640,
  "nfeatures": 600,
  "ratio": 0.75,
  "ransac_px": 4.0,
  "mask": [
   0.2,
   0.15,
   0.8,
   0.95
  ]
 },
 "near_dup_sift_inliers": 20,
 "near_dup_geo_stone_l1": 0.06,
 "near_dup_cos": 0.97,
 "near_dup_vis_stone_l1": 0.05,
 "template_exemplars": [
  "amethyst/gp106340-4-191225",
  "amethyst/gp160323-4-20626",
  "blue_sapphire/gp119501-4-120126"
 ],
 "template_cos": 0.9,
 "max_group_images": 12,
 "split_tolerance_pts": 3.0
}
```

Versions: {'python': '3.11.16', 'numpy': '2.4.6', 'pandas': '3.0.6', 'torch': '2.14.1', 'sklearn': '1.9.1'}

CSV SHA-256: {'train.csv': '903915cd2b66aa5711db6eebe05abeeac80e84f284b85501e7f2102e813774f3', 'validation.csv': '861595181bba7b877ae510771a495d8a5a86493de87802d61d8a5b662bfea65f', 'test.csv': '594987219e753901116bafe03b551d33fca7c943c63d119c95cea579c74f52e5'}