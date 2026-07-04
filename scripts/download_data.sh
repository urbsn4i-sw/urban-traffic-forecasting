#!/usr/bin/env bash
# download_data.sh — 교통 STGNN 데이터 취득 안내 + 배치 검증
# =====================================================================
# METR-LA / PEMS-BAY 는 연구용 공개 데이터지만, 원 배포가 Google Drive(용량 큼)라
# 이 스크립트는 "무엇도 자동으로 내려받지 않는다".
#   - (1) 취득 출처/라이선스 문구를 출력하고
#   - (2) 로컬에 이미 배치된 파일이 있으면 "존재 여부만" 읽기전용으로 점검한다.
# 임의 URL 실행·자동 wget/curl 다운로드는 금지(CLAUDE.md 절대규칙 2, data-and-hub 규칙).
#
# 기본 경로: --subset metr-la  (다른 값: pems-bay)
# 데이터는 DATA_ROOT(기본 data/ (repo root)) 아래에 두며 .gitignore 로 커밋 차단된다.
# =====================================================================
set -euo pipefail

SUBSET="metr-la"
DATA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
FETCH=0
# DCRNN(liyaguang/DCRNN) README 가 안내하는 공식 Google Drive 폴더 id (metr-la.h5·pems-bay.h5 포함)
DCRNN_DRIVE_FOLDER="10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX"

usage() {
  cat <<EOF
사용법: bash download_data.sh [--subset metr-la|pems-bay] [--data-root PATH] [--fetch]

  --subset     metr-la(기본, 207센서) | pems-bay(325센서)
  --data-root  데이터 저장 위치 (기본: data/ (repo root))
  --fetch      (opt-in) 공식 DCRNN Google Drive 폴더에서 gdown 으로 .h5 취득 시도.
               연구용 공개 데이터에 한함. 기본은 안내+배치검증만(자동 다운로드 안 함).

--fetch 없이는 자동 다운로드를 하지 않는다(안내 + 배치 검증만).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subset) SUBSET="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --fetch) FETCH=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "알 수 없는 인자: $1"; usage; exit 1 ;;
  esac
done

# opt-in 취득: 공식 DCRNN Drive 폴더에서 gdown 으로 .h5 내려받기(연구용 공개 데이터).
# 재현 기록: Phase 1 에서 실제 이 방법(gdown --folder <공식 id>)으로 metr-la.h5 취득 성공.
fetch_gdown() {
  echo "[fetch] 공식 DCRNN Google Drive 폴더에서 gdown 으로 취득 시도(연구용 공개 데이터)."
  if ! python -c "import gdown" >/dev/null 2>&1; then
    echo "[fetch] gdown 미설치 → 'pip install gdown' 후 재실행하거나, 브라우저로 직접 취득해 배치하세요." >&2
    return 1
  fi
  mkdir -p "${DATA_ROOT}"
  ( cd "${DATA_ROOT}" && python -m gdown --folder \
      "https://drive.google.com/drive/folders/${DCRNN_DRIVE_FOLDER}" -O . )
  echo "[fetch] 완료(또는 부분). 아래 배치 검증으로 결과 확인. 받은 .h5 는 .gitignore 로 커밋 차단됨."
}

echo "[download] subset=${SUBSET}  data_root=${DATA_ROOT}"
mkdir -p "${DATA_ROOT}"

# ---------------------------------------------------------------------
# 1) 라이선스 / 출처 고지
# ---------------------------------------------------------------------
print_tou() {
  cat <<'TOU'
────────────────────────────────────────────────────────────────────
[출처·라이선스 고지 — 취득 전 확인]

■ METR-LA / PEMS-BAY (Li et al. 2018, DCRNN)
  - 성격: 연구/학습용 공개 교통 속도 데이터.
    · METR-LA : LA 고속도로 루프검출기 207센서, 5분 간격(2012.03~06).
    · PEMS-BAY: Bay Area 325센서, 5분 간격(Caltrans PeMS 파생).
  - 취득: DCRNN 공개 저장소(github.com/liyaguang/DCRNN) README 의 Google Drive/Baidu 링크에서
    아래 파일을 직접 내려받아 배치한다(이 스크립트가 대신 받지 않음):
      · metr-la.h5 / pems-bay.h5       (센서×시간 속도 행렬)
      · adj_mx.pkl / adj_mx_bay.pkl    (센서 간 도로망 거리 기반 인접행렬)
      · (선택) distances/sensor_ids csv (인접행렬을 직접 만들 때)
  - 인용: Li, Yu, Shahabi, Liu (2018) DCRNN, ICLR. 원 PeMS 데이터는 Caltrans 제공.

※ 위 .h5/.pkl 및 파생 .npz 는 이 저장소에 절대 커밋하지 않는다(.gitignore 로 차단).
※ 아래 안내는 "공식 배포처로 이동해 직접 취득"하라는 것이며, 스크립트가 다운로드하지 않는다.
────────────────────────────────────────────────────────────────────
TOU
}

# ---------------------------------------------------------------------
# 2) 기대 디렉토리 레이아웃
# ---------------------------------------------------------------------
print_layout() {
  cat <<EOF
[기대 레이아웃] (수동 취득 후 아래처럼 배치)
  ${DATA_ROOT}/
  ├─ metr-la.h5            # (subset=metr-la) 속도 행렬 (T × 207)
  ├─ pems-bay.h5           # (subset=pems-bay) 속도 행렬 (T × 325)
  ├─ adj_mx.pkl            # METR-LA 인접행렬(pickle: sensor_ids, id_map, adj)
  ├─ adj_mx_bay.pkl        # PEMS-BAY 인접행렬
  └─ processed/            # (Phase 1 에서 생성) train/val/test .npz 윈도우
EOF
}

# ---------------------------------------------------------------------
# 3) 배치 검증 (읽기 전용) — 있으면 OK, 없으면 안내만.
# ---------------------------------------------------------------------
check_present() {
  echo "[check] 로컬 배치 상태 점검 (읽기 전용):"
  local ok=0
  local h5 adj
  if [[ "${SUBSET}" == "pems-bay" ]]; then
    h5="pems-bay.h5"; adj="adj_mx_bay.pkl"
  else
    h5="metr-la.h5"; adj="adj_mx.pkl"
  fi
  for f in "${h5}" "${adj}"; do
    if [[ -f "${DATA_ROOT}/${f}" ]]; then
      echo "  ✓ ${DATA_ROOT}/${f}"
    else
      echo "  x 없음: ${DATA_ROOT}/${f}  (수동 취득·배치 필요)"
      ok=1
    fi
  done
  return "${ok}"
}

# ---------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------
print_tou
echo ""
print_layout
echo ""
if [[ "${FETCH}" -eq 1 ]]; then
  fetch_gdown || echo "[fetch] 자동 취득 실패/미완 — 정직하게 보고하고 수동 취득으로 진행하세요."
  echo ""
fi
if check_present; then
  echo "[check] 필수 파일이 존재합니다. (Phase 1 윈도우 생성/학습으로 진행 가능)"
else
  echo "[check] 일부/전부 미배치. 위 출처에서 수동 취득해 DATA_ROOT 에 배치하세요."
fi

cat <<'NOTE'

[download] 요약: 이 스크립트는 자동 다운로드를 수행하지 않았습니다.
  → .h5/.pkl 은 DCRNN 공개 저장소의 배포 링크에서 직접 취득해 DATA_ROOT 에 배치하세요.
  → train/val/test .npz 윈도우 생성은 Phase 1 실행 단계에서 승인 후 진행합니다(대용량, 커밋 금지).
NOTE
