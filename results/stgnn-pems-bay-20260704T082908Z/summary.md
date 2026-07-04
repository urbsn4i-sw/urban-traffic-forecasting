# PEMS-BAY STGNN + 기준선 — stgnn-pems-bay-20260704T082908Z

- 데이터 PEMS-BAY (T,N)=(52116,325), test 10419 윈도우, 원 단위(mph), masked(null=0)
- 프로토콜 과거 12→미래 12 스텝, 특징=[zscore speed, time-of-day]
- device=cuda, seed=42, epochs_cap=50, batch=256, git=693178be
- SOTA 재현 아님(원리 재현). 논문값은 참조용(설정 차이로 직접 비교 아님).

| 모델 | MAE@15m | MAE@30m | MAE@60m | RMSE@60m | MAPE@60m(%) | MAE기울기 | 비고 |
|---|---|---|---|---|---|---|---|
| copy-last | 1.598 | 2.179 | 3.048 | 7.015 | 6.83 | 0.1785 | 기준선 |
| seasonal-HA | 3.029 | 3.027 | 3.024 | 5.841 | 7.12 | -0.0005 | 기준선 |
| STGNN (fixed) | 1.475 | 1.980 | 2.647 | 5.902 | 6.44 | 0.1473 | 771.0s·2.22GB·ep45 |
| STGNN (learned) | 1.433 | 1.866 | 2.400 | 5.252 | 5.99 | 0.1244 | 778.8s·1.63GB·ep50 |
| STGNN (hybrid) | 1.431 | 1.853 | 2.368 | 5.217 | 5.85 | 0.1209 | 799.5s·1.63GB·ep43 |
| STGNN (identity) | 1.505 | 2.055 | 2.861 | 6.536 | 6.91 | 0.1668 | 572.9s·1.62GB·ep27 |
| *(참조)* *DCRNN 논문* | *1.38* | *1.74* | *2.07* | *—* | *—* | *—* | *참조용* |

> RQ1(고정 vs 학습): 위 STGNN 행들의 fixed vs learned/hybrid MAE 비교.
> RQ2(오차 누적): MAE기울기(스텝당) — 기준선 copy-last 대비 STGNN 이 덜 누적하는지.
