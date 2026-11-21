# Results

_Generated 2026-11-21T19:48:53+00:00_


## Test split (real signals, unseen stars)

| Model | Macro PR-AUC | Macro F1 | Accuracy | PC recall @95% prec. | PC F1 | EB F1 | FP F1 | ECE (PC) | n |
|---|---|---|---|---|---|---|---|---|---|
| gbm | 0.870 | 0.805 | 0.804 | 0.129 | 0.800 | 0.885 | 0.730 | 0.074 | 92 |
| cnn | 0.812 | 0.705 | 0.707 | 0.258 | 0.667 | 0.812 | 0.635 | 0.087 | 92 |


## Test split: Kepler stars only

| Model | Macro PR-AUC | Macro F1 | Accuracy | PC recall @95% prec. | PC F1 | EB F1 | FP F1 | ECE (PC) | n |
|---|---|---|---|---|---|---|---|---|---|
| gbm | 0.858 | 0.791 | 0.789 | 0.125 | 0.773 | 0.870 | 0.731 | 0.087 | 71 |
| cnn | 0.801 | 0.689 | 0.690 | 0.375 | 0.667 | 0.760 | 0.640 | 0.076 | 71 |


## Test split: TESS stars only

| Model | Macro PR-AUC | Macro F1 | Accuracy | PC recall @95% prec. | PC F1 | EB F1 | FP F1 | ECE (PC) | n |
|---|---|---|---|---|---|---|---|---|---|
| gbm | 0.892 | 0.845 | 0.857 | 0.143 | 0.875 | 0.933 | 0.727 | 0.150 | 21 |
| cnn | 0.863 | 0.761 | 0.762 | 0.286 | 0.667 | 1.000 | 0.615 | 0.205 | 21 |


## Validation split

| Model | Macro PR-AUC | Macro F1 | Accuracy | PC recall @95% prec. | PC F1 | EB F1 | FP F1 | ECE (PC) | n |
|---|---|---|---|---|---|---|---|---|---|
| gbm | 0.824 | 0.756 | 0.764 | 0.167 | 0.787 | 0.885 | 0.596 | 0.107 | 106 |
| cnn | 0.759 | 0.748 | 0.745 | 0.167 | 0.747 | 0.862 | 0.635 | 0.096 | 106 |


### Confusion matrix: gbm (test)

| true \ pred | PC | EB | FP |
|---|---|---|---|
| PC | 24 | 1 | 6 |
| EB | 0 | 27 | 2 |
| FP | 5 | 4 | 23 |


gbm CPU latency: 1.27 ms per candidate


### Confusion matrix: cnn (test)

| true \ pred | PC | EB | FP |
|---|---|---|---|
| PC | 19 | 3 | 9 |
| EB | 1 | 26 | 2 |
| FP | 6 | 6 | 20 |


cnn CPU latency: 1.05 ms per candidate


Dataset: 943 examples from 846 stars (0 synthetic), built 2026-11-21T18:32:41+00:00.
