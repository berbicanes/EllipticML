| Model        | Features          |   n_features |    F1 |   PR-AUC |   F1 diff vs AF |    F1 p |   PR-AUC diff vs AF |   PR-AUC p |
|:-------------|:------------------|-------------:|------:|---------:|----------------:|--------:|--------------------:|-----------:|
| RandomForest | AF                |          165 | 0.804 |    0.793 |           0.000 | nan     |               0.000 |    nan     |
| RandomForest | AF without clocks |          160 | 0.801 |    0.793 |           0.000 |   1.000 |               0.000 |      0.842 |
| XGBoost      | AF                |          165 | 0.794 |    0.798 |           0.000 | nan     |               0.000 |    nan     |
| XGBoost      | AF without clocks |          160 | 0.800 |    0.799 |          -0.000 |   0.855 |               0.001 |      0.264 |
