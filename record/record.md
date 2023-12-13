## Fishyscapes val

### DeepLabV3+

#### Baselines 

Sigmoid-training

| AP | FPR@95 | AUROC| mIOU |
|:-:|:-:|:-:|:-:|
| 3.5 | 76.7 | 81.2 | 71.2|

MSP 

| AP | FPR@95 | AUROC| mIOU |
|:-:|:-:|:-:|:-:|
| 11.2 | 41.5 | 89.5 | 79.3|

Max Logit

| AP | FPR@95 | AUROC| mIOU |
|:-:|:-:|:-:|:-:|
| 24.9 | 29.8 | 94.5 | 79.3|

Energy

| AP | FPR@95 | AUROC| mIOU |
|:-:|:-:|:-:|:-:|
| 28.0 | 29.6 | 94.8 | 79.3|

Margin
| AP | FPR@95 | AUROC| mIOU |
|:-:|:-:|:-:|:-:|
| 2.1 | 42.9 | 87.8 | 79.3|

#### Patch OOD

copy&past

| AP | FPR@95 | AUROC| mIOU |
|:-:|:-:|:-:|:-:|
| 7.7 | 51.7 | 89.3 | 79.3|

copy&past + Max Logit

| AP | FPR@95 | AUROC| mIOU |
|:-:|:-:|:-:|:-:|
| 34.8 | 28.9 | 94.9 | 79.3|

copy&past (1.0) + energy

| AP | FPR@95 | AUROC| mIOU |
|:-:|:-:|:-:|:-:|
| 41.3 | 28.4 | 95.3 | 79.3|

copy&past (2.0) + energy

| AP | FPR@95 | AUROC| mIOU |
|:-:|:-:|:-:|:-:|
| 44.4 | 27.3 | 95.5 | 79.3|

copy&past-nosoftmax (0.5) + energy

| AP | FPR@95 | AUROC| mIOU |
|:-:|:-:|:-:|:-:|
| 44.7 | 25.4 | 95.9 | 79.3|

copy&past-nosoftmax-noise (0.5) + energy

| AP | FPR@95 | AUROC| mIOU |
|:-:|:-:|:-:|:-:|
| 28.9 | 26.7 | 95.3 | 79.3|

copy&past-nosoftmax-noise (0.5) + energy

| AP | FPR@95 | AUROC| mIOU |
|:-:|:-:|:-:|:-:|
| 45.8 | 26.6 | 95.8 | 79.3|