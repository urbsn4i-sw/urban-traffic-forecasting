# METR-LA STGNN + 기준선 — stgnn-metr-la-20260704T064753Z

- 데이터 METR-LA (T,N)=(34272,207), test 6850 윈도우, 원 단위(mph), masked(null=0)
- 프로토콜 과거 12→미래 12 스텝, 특징=[zscore speed, time-of-day]
- device=cuda, seed=42, epochs_cap=50, batch=256, git=9baa6e1d
- SOTA 재현 아님(원리 재현). 논문값은 참조용(설정 차이로 직접 비교 아님).

| 모델 | MAE@15m | MAE@30m | MAE@60m | RMSE@60m | MAPE@60m(%) | MAE기울기 | 비고 |
|---|---|---|---|---|---|---|---|
| copy-last | 4.017 | 5.094 | 6.795 | 14.209 | 16.71 | 0.3321 | 기준선 |
| seasonal-HA | 4.187 | 4.187 | 4.187 | 7.852 | 13.03 | 0.0000 | 기준선 |
| STGNN (fixed) | 3.112 | 3.795 | 4.889 | 9.500 | 14.38 | 0.2124 | 319.6s·1.38GB·ep50 |
| STGNN (learned) | 2.998 | 3.497 | 4.273 | 8.290 | 13.07 | 0.1543 | 321.4s·1.0GB·ep49 |
| STGNN (hybrid) | 2.998 | 3.516 | 4.298 | 8.318 | 13.03 | 0.1587 | 336.2s·1.0GB·ep50 |
| STGNN (identity) | 3.147 | 3.841 | 4.951 | 9.708 | 14.72 | 0.2152 | 321.4s·1.0GB·ep50 |
| *(참조)* *DCRNN 논문* | *2.77* | *3.15* | *3.60* | *—* | *—* | *—* | *참조용* |

> RQ1(고정 vs 학습): 위 STGNN 행들의 fixed vs learned/hybrid MAE 비교.
> RQ2(오차 누적): MAE기울기(스텝당) — 기준선 copy-last 대비 STGNN 이 덜 누적하는지.
