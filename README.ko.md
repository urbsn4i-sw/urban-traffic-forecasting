# urban-traffic-forecasting — METR-LA & PEMS-BAY 교통 STGNN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 🇬🇧 **[English README](README.md)**

도시 교통 **시공간 예측**을 그래프 신경망(STGNN)으로 게이밍 PC(**RTX 3060, 8GB**)에서 **정직하게
축소 재현**한다. **두 도시 — METR-LA(LA)·PEMS-BAY(Bay Area)** 도로 센서 속도의 미래를 예측하며,
**인접행렬을 학습하면 이득인가**(RQ1, 두 도시에서 검증)와 **다단계 예측에서 오차가 어떻게
누적되는가**(RQ2)를 실측으로 확인한다. 목적은 SOTA 재현이 아니라 **원리 재현·학습**이다.

## 핵심 원칙 — 결과를 지어내지 않는다
실제 실행값(`metrics.json`)만 보고한다. 안 돌린 것은 **한계**로 명시한다. 원데이터·인접행렬·가중치는
**절대 커밋하지 않으며**(`.gitignore` 차단), `scripts/download_data.sh` 가 데이터 취득 방법을 안내한다.

## 파이프라인
```mermaid
flowchart LR
  A["METR-LA(207) · PEMS-BAY(325)<br/>5분 속도<br/>(로컬, 미커밋)"] --> B["전처리<br/>윈도우 12→12 · z-score<br/>시간순 70/10/20"]
  B --> C["기준선<br/>copy-last · seasonal-HA"]
  B --> D["소형 STGNN<br/>확산 그래프합성곱 + 게이트드 TCN"]
  D --> E["인접행렬 절제 (RQ1)<br/>fixed · learned · hybrid · identity"]
  C --> F["평가 (test)<br/>MAE/RMSE/MAPE @15/30/60분"]
  E --> F
  F --> G["metrics.json"]:::done

  classDef done fill:#c6f6d5,stroke:#2f855a,color:#1a202c;
```

## 앵커 · 브리지 대응 (GraphCast ↔ 교통 STGNN)
| GraphCast (Lam+ 2023, 지구 기상) | 교통 STGNN (본 저장소) |
|---|---|
| 지구 격자 → 다중메시 **그래프** | 도로 센서 → **그래프**(거리 기반 엣지) |
| 노드 특징 = 기상 상태 | 노드 특징 = 센서 속도(+ time-of-day) |
| 메시 위 GNN 메시지 패싱 | 확산 그래프합성곱(DCRNN) / adaptive 그래프(Graph WaveNet) |
| 6시간 한 스텝 후 **자기회귀 롤아웃** | 다단계 15/30/60분 예측, **오차 누적** 관찰(RQ2) |
| 고정 메시(지구 기하) | **고정 vs 학습** 인접행렬 절제(RQ1) |

## 재현 방법
```bash
# 0) 환경 (RTX 3060 / CUDA; CPU 폴백 가능)
pip install -r requirements.txt
#   GPU torch: pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
#   (Windows+Anaconda: 스크립트가 KMP_DUPLICATE_LIB_OK=TRUE 설정 — OpenMP DLL 충돌 회피)

# 1) 데이터 (연구용 공개; 자동 다운로드 안 함 — 스크립트 안내)
pip install gdown
bash scripts/download_data.sh --subset metr-la --fetch   # metr-la.h5 / pems-bay.h5 -> data/ (gitignore)

# 2) 스모크 테스트 (합성 텐서; 데이터·모델 불필요, numpy-only)
bash scripts/smoke.sh

# 3) 기준선 (실 test split) — 데이터셋 선택
python scripts/eval_baselines.py --dataset metr-la      # 또는 --dataset pems-bay

# 4) STGNN 학습 + 인접행렬 절제(RQ1) — RTX 3060
python scripts/train_stgnn.py --dataset metr-la  --modes fixed learned hybrid identity --epochs 50 --batch-size 256  # ~5분/모드
python scripts/train_stgnn.py --dataset pems-bay --modes fixed learned hybrid identity --epochs 50 --batch-size 256  # ~13분/모드 (325센서)
#   -> results/<run_id>/{metrics.json, summary.md}
```

## 실측 결과 (실제 실행값) — 두 도시
**두 데이터셋, 같은 파이프라인·같은 조건**(2-layer STGNN, hidden 32, 특징 `[z-score speed,
time-of-day]`, seed 42, ≤50 epochs, batch 256, RTX 3060). 시간순 **70/10/20**, **test split**,
masked MAE/RMSE/MAPE(결측=0 제외), **원 단위(mph)**.

### METR-LA (207센서 · 2012 · 결측 ~8%) — test 6,850 윈도우
| 모델 | MAE@15m | MAE@30m | MAE@60m | RMSE@60m | MAPE@60m (%) | MAE 기울기/스텝 |
|---|---|---|---|---|---|---|
| copy-last (persistence) | 4.017 | 5.094 | 6.795 | 14.209 | 16.71 | 0.332 |
| seasonal-HA (DCRNN 정의) | 4.187 | 4.187 | 4.187 | 7.852 | 13.03 | ~0.000 |
| STGNN **fixed** (도로망 A) | 3.112 | 3.795 | 4.889 | 9.500 | 14.38 | 0.212 |
| STGNN **learned** (adaptive A) | **2.998** | **3.497** | **4.273** | **8.290** | **13.07** | 0.154 |
| STGNN **hybrid** (fixed+adaptive) | 2.998 | 3.516 | 4.298 | 8.318 | 13.03 | 0.159 |
| STGNN **identity** (그래프 없음) | 3.147 | 3.841 | 4.951 | 9.708 | 14.72 | 0.215 |
| *(참조)* *DCRNN / HA 논문* | *2.77 / 4.16* | *3.15 / 4.16* | *3.60 / 4.16* | *—* | *—* | *—* |

전체: [`results/stgnn-metr-la-…/metrics.json`](results/stgnn-metr-la-20260704T064753Z/metrics.json).

### PEMS-BAY (325센서 · 2017 · 결측 ~0%) — test 10,419 윈도우
| 모델 | MAE@15m | MAE@30m | MAE@60m | RMSE@60m | MAPE@60m (%) | MAE 기울기/스텝 |
|---|---|---|---|---|---|---|
| copy-last (persistence) | 1.598 | 2.179 | 3.048 | 7.015 | 6.83 | 0.179 |
| seasonal-HA (DCRNN 정의) | 3.029 | 3.027 | 3.024 | 5.841 | 7.12 | ~0.000 |
| STGNN **fixed** (도로망 A) | 1.475 | 1.980 | 2.647 | 5.902 | 6.44 | 0.147 |
| STGNN **learned** (adaptive A) | 1.433 | 1.866 | 2.400 | 5.252 | 5.99 | 0.124 |
| STGNN **hybrid** (fixed+adaptive) | **1.431** | **1.853** | **2.368** | **5.217** | **5.85** | 0.121 |
| STGNN **identity** (그래프 없음) | 1.505 | 2.055 | 2.861 | 6.536 | 6.91 | 0.167 |
| *(참조)* *DCRNN / HA 논문* | *1.38 / 2.88* | *1.74 / 2.88* | *2.07 / 2.88* | *—* | *—* | *—* |

> PEMS-BAY 는 결측이 거의 0이고 고속도로라 교통이 매끄러워 **모든 오차가 METR-LA 보다 훨씬 낮다**
> (문헌과 일치). 전체: [`results/stgnn-pems-bay-…/metrics.json`](results/stgnn-pems-bay-20260704T082908Z/metrics.json).
> ⚠️ 논문값은 **참조용**(우리 축소 2-layer·≤50ep 설정과 달라 직접 비교 아님) — 우리 결과로 옮겨 적지 않는다.

### RQ1 — 인접행렬 고정 vs 학습 (**두 도시 모두** 성립)
60분 MAE 기준 순서가 **두 데이터셋에서 동일**: `learned ≈ hybrid` **<** `fixed`(도로망) **<** `identity`(그래프 없음).
- **METR-LA:** learned 4.27 ≈ hybrid 4.30 < fixed 4.89 < identity 4.95
- **PEMS-BAY:** hybrid 2.37 ≈ learned 2.40 < fixed 2.65 < identity 2.86

**답: 데이터로 인접행렬을 학습하는 편이 고정 도로망 그래프보다 낫고, 그래프가 있는 편이 없는 것
(identity 최악)보다 낫다 — 두 도시 모두에서.** `hybrid` ≈ `learned`(결합이 학습 단독을 유의하게
넘지 못함)도 두 도시 공통. RQ1 결과가 **두 도시에서 일반화**된다.

### STGNN이 기준선을 이기는가
- **METR-LA:** copy-last 는 전 지평 우세; seasonal-HA 는 15/30분 우세이나 **60분은 HA 근소 우위**
  (4.19 vs learned 4.27) — 간선도로 교통이라 주간 계절성이 경쟁력.
- **PEMS-BAY:** **두 기준선 모두를 전 지평에서** 이김(learned 1.43/1.87/2.40 vs copy-last 1.60/2.18/3.05,
  seasonal-HA ~3.03). 매끄러운 고속도로 데이터에서 seasonal-HA 는 약한 기준선.

### RQ2 — 다단계 오차 누적
스텝당 MAE 기울기(STGNN-learned vs copy-last): **METR-LA 0.154 vs 0.332**(약 절반), **PEMS-BAY
0.124 vs 0.179**(~70%). 두 도시 모두 STGNN 이 persistence 보다 오차를 천천히 누적. seasonal-HA 는
구조상 평탄(~0).

## 한계 / 안 한 것 (정직하게)
- **SOTA 아님(원리 재현):** 우리 소형 모델은 논문값(METR-LA DCRNN 2.77/3.15/3.60; PEMS-BAY
  1.38/1.74/2.07)에 못 미친다. 두 도시에서 RQ1/RQ2 확인이 목적.
- **METR-LA 60분 지평:** seasonal-HA 가 근소 우위(PEMS-BAY 에선 STGNN 이 전 지평 우세). 더 큰 모델·
  긴 학습이면 METR-LA@60m 이 뒤집힐 여지가 있으나 이번 설정에선 미달.
- **GPU 비결정성:** cudnn.deterministic 이 이 Conv1d 에서 ~25배 느려 benchmark 모드로 학습. 시드는
  고정(초기화·데이터 순서)이나 GPU 합성곱은 bitwise 재현 아님.
- **범위:** 두 데이터셋(METR-LA·PEMS-BAY); 더 큰 STGNN·더 긴 지평은 후속 과제.
- `smoke.sh` 수치는 합성(`synthetic_dummy=true`)이며 성능이 아니다.

## 데이터 출처·라이선스
- **코드(이 저장소): MIT** — [`LICENSE`](LICENSE) 참조.
- **데이터는 미포함이며 MIT 대상 아님:** METR-LA / PEMS-BAY 는 **연구용 공개** 교통 데이터로
  [DCRNN 저장소](https://github.com/liyaguang/DCRNN)(Google Drive)에서 배포되며, 원 루프검출기
  데이터는 Caltrans PeMS 제공. `scripts/download_data.sh` 로 직접 취득한다. 이 저장소에는
  **원데이터·가중치가 없다**(`.gitignore`).

## References (인용)
BibTeX: [`CITATIONS.md`](CITATIONS.md).
- **[앵커]** Lam, R., et al. (2023). *Learning skillful medium-range global weather forecasting.*
  **Science, 382, 1416–1421.** DOI [10.1126/science.adi2336](https://doi.org/10.1126/science.adi2336) [SCI(E)].
  — GraphCast: 여기서 도시 규모로 대응하는 격자→그래프→롤아웃 패러다임.
- **[브리지]** Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018). *DCRNN: Data-Driven Traffic Forecasting.*
  **ICLR 2018.** arXiv [1707.01926](https://arxiv.org/abs/1707.01926).
  — 확산 그래프합성곱·METR-LA/PEMS-BAY 벤치마크·HA 기준선 정의.
- **[브리지]** Wu, Z., et al. (2019). *Graph WaveNet for Deep Spatial-Temporal Graph Modeling.*
  **IJCAI 2019.** DOI [10.24963/ijcai.2019/264](https://doi.org/10.24963/ijcai.2019/264).
  — adaptive(학습) 인접행렬·게이트드 시간 합성곱.

## Contact / Author
- **Author:** urbsn4i-sw (GitHub). 학습·재현 목적의 비상업 저장소.
- 문의·재현 이슈는 GitHub Issues 로.
