# Image trace

Rebuilt by `code/data_trace.py` from the dataset files. Counts are images unless stated.

| stage                                              | images   | stones   | note                                                                             |
|:---------------------------------------------------|:---------|:---------|:---------------------------------------------------------------------------------|
| Earlier paper, Table 1 (8 classes)                 | 3,182    |          | split 2,545 train / 637 validation, no test set; image files not retained        |
| New scrape: gallery photos listed                  | 8,798    |          | 1,829 products, 11 classes                                                       |
| Excluded by role (hand shots, certificates, other) | −3,264   |          | 1,523 hand shots, 1,732 certificates, 9 other                                    |
| Loose-stone photos downloaded                      | 5,534    |          | one product page per stone, numbered stone views only                            |
| Cleaning and deduplication                         | −8       |          | 5 promo banner images, 3 exact duplicates within one product; 0 corrupt          |
| After cleaning (11 classes)                        | 5,526    | 1,790    | near-duplicate links across products (44) merged into stone groups, none deleted |
| White Sapphire removed                             | −502     | −161     | 97% black backdrop; see white_sapphire_exclusion.md                              |
| Benchmark (10 classes)                             | 5,024    | 1,629    | train 3,518 / validation 755 / test 751                                          |


## Final 10-class split

| class           |   images |   stones |   train |   validation |   test |
|:----------------|---------:|---------:|--------:|-------------:|-------:|
| Peridot         |      502 |      167 |     351 |           75 |     76 |
| Ametrine        |      501 |      171 |     351 |           75 |     75 |
| Blue Zircon     |      502 |      167 |     352 |           75 |     75 |
| Yellow Sapphire |      502 |      164 |     351 |           76 |     75 |
| Ruby            |      521 |      127 |     364 |           79 |     78 |
| Emerald         |      501 |      153 |     351 |           75 |     75 |
| Blue Sapphire   |      499 |      160 |     350 |           75 |     74 |
| Pink Sapphire   |      497 |      167 |     348 |           75 |     74 |
| Amethyst        |      497 |      172 |     348 |           75 |     74 |
| Citrine         |      502 |      181 |     352 |           75 |     75 |
| Total           |     5024 |     1629 |    3518 |          755 |    751 |



## All 11 classes, before White Sapphire was removed

| class           |   products |   listed |   hand_shots |   certificates |   other_files |   downloaded |   removed_promo |   removed_duplicate |   kept |   stones |   train |   validation |   test |   earlier_paper |
|:----------------|-----------:|---------:|-------------:|---------------:|--------------:|-------------:|----------------:|--------------------:|-------:|---------:|--------:|-------------:|-------:|----------------:|
| Peridot         |        173 |      777 |          102 |            173 |             0 |          502 |               0 |                   0 |    502 |      167 |     351 |           75 |     76 |             480 |
| Ametrine        |        179 |      784 |          104 |            178 |             1 |          501 |               0 |                   0 |    501 |      171 |     351 |           75 |     75 |             262 |
| Blue Zircon     |        174 |      785 |          107 |            174 |             2 |          502 |               0 |                   0 |    502 |      167 |     352 |           75 |     75 |             246 |
| White Sapphire  |        163 |      817 |          160 |            154 |             1 |          502 |               0 |                   0 |    502 |      161 |     352 |           75 |     75 |             466 |
| Yellow Sapphire |        164 |      803 |          156 |            145 |             0 |          502 |               0 |                   0 |    502 |      164 |     351 |           76 |     75 |             442 |
| Ruby            |        127 |      760 |          116 |            122 |             1 |          521 |               0 |                   0 |    521 |      127 |     364 |           79 |     78 |             444 |
| Emerald         |        154 |      796 |          154 |            141 |             0 |          501 |               0 |                   0 |    501 |      153 |     351 |           75 |     75 |             446 |
| Blue Sapphire   |        160 |      786 |          140 |            146 |             0 |          500 |               1 |                   0 |    499 |      160 |     350 |           75 |     74 |             396 |
| Pink Sapphire   |        168 |      804 |          157 |            144 |             3 |          500 |               0 |                   3 |    497 |      167 |     348 |           75 |     74 |             nan |
| Amethyst        |        180 |      841 |          171 |            168 |             1 |          501 |               4 |                   0 |    497 |      172 |     348 |           75 |     74 |             nan |
| Citrine         |        187 |      845 |          156 |            187 |             0 |          502 |               0 |                   0 |    502 |      181 |     352 |           75 |     75 |             nan |

