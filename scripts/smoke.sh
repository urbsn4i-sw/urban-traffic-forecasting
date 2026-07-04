#!/usr/bin/env bash
# smoke.sh — 교통 STGNN 파이프라인 스모크런 (더미 입력 기준선→지표)
# =====================================================================
# 목적: 실 데이터/모델 없이, 소량 합성 교통 시계열 1회로
#       "기준선 예측 → 지표(MAE/RMSE/MAPE @15/30/60분) → metrics.json 저장"
#       파이프라인이 끝까지 도는지만 확인. (numpy 만, torch 불필요)
# ⚠️ METR-LA/PEMS-BAY 데이터도, STGNN 가중치도 받지도·쓰지도 않는다. 성능 보고가 아니다.
#
# 실 데이터 기반 학습/평가는 데이터 준비(Phase 1) 후 train.py/eval.py 로 붙인다.
# =====================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"

# 출력은 커밋하지 않는 임시 디렉토리로(결과가 results/ 에 남지 않게).
OUT_DIR="$(mktemp -d 2>/dev/null || echo "${TMPDIR:-/tmp}/traffic_smoke_$$")"
mkdir -p "${OUT_DIR}"
trap 'rm -rf "${OUT_DIR}"' EXIT

echo "[smoke] 더미 교통 기준선→지표 파이프라인 실행 (out=${OUT_DIR})"
"${PY}" "${SCRIPT_DIR}/run_baseline_demo.py" \
  --seed 42 --t-in 12 --horizon 12 --num-nodes 8 --num-steps 864 \
  --out "${OUT_DIR}"

# 산출물 존재 검증
if [[ -f "${OUT_DIR}/metrics.json" ]]; then
  echo "[smoke] OK: metrics.json 생성 확인. 파이프라인 정상."
else
  echo "[smoke] FAIL: metrics.json 이 없습니다." >&2
  exit 1
fi
