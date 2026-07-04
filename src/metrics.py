"""교통 STGNN 표준 지표 구현 — MAE / RMSE / MAPE (masked).

과제별 표준 지표(PROJECT_GUIDELINE.md §8): **MAE · RMSE · MAPE @ horizon 3/6/12**
(= 15/30/60분, 5분 간격 기준).

설계 원칙 (common/metrics.py 와 동일)
  - numpy 외 무거운 의존성 없음. numpy 는 함수 내부 지연 임포트(Phase 0 미설치 환경에서도
    모듈 import 만으로 깨지지 않게).
  - **결과를 지어내지 않는다**: 유효(관측)값이 하나도 없으면 임의 기본값 대신 NaN 을 돌려준다.
  - py3.8 호환.

교통 지표의 결측 처리 규약 (DCRNN / Graph WaveNet 관례)
  - METR-LA / PEMS-BAY 는 **결측 관측을 0 으로 채워** 배포한다. 따라서 지표는 `null_val`
    (기본 0.0) 인 위치를 **마스크로 제외**하고 계산한다. 0 을 정상값으로 두면 MAPE 가 발산한다.
  - `null_val=nan` 을 주면 NaN 위치를 결측으로 본다.

형상 규약
  - 단일 지표 함수: pred / gt 는 임의 형상(브로드캐스트 동일). 마스크 후 평면 평균.
  - 지평별 함수: (T_out, *) — axis 0 이 예측 지평 스텝. horizon h → 인덱스 h-1.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence


def _np():
    """numpy 지연 임포트 (미설치 환경에서 모듈 import 는 되게)."""
    import numpy as np  # noqa: PLC0415
    return np


# ---------------------------------------------------------------------
# 저장 유틸 (커밋 허용 산출물) — common.metrics.save_metrics 와 동일 계약
# ---------------------------------------------------------------------
def save_metrics(metrics: dict[str, Any], out_path: str | Path) -> Path:
    """지표 dict 를 results/<run>/metrics.json 으로 저장."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# ---------------------------------------------------------------------
# 결측 마스크
# ---------------------------------------------------------------------
def _valid_mask(gt, null_val):
    """관측 유효(=결측 아님) 위치 불리언 마스크. null_val=nan 이면 비-NaN 을 유효로 본다."""
    np = _np()
    gt = np.asarray(gt, dtype=np.float64)
    if null_val is None:
        return np.ones(gt.shape, dtype=bool)
    if isinstance(null_val, float) and np.isnan(null_val):
        return ~np.isnan(gt)
    return gt != null_val


def _check_shapes(pred, gt):
    np = _np()
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt 형상 불일치: {pred.shape} vs {gt.shape}")
    return pred, gt


# ---------------------------------------------------------------------
# 단일 지표 (masked)
# ---------------------------------------------------------------------
def masked_mae(pred: Any, gt: Any, null_val: float = 0.0) -> float:
    """평균 절대 오차. 유효값 없으면 NaN."""
    np = _np()
    pred, gt = _check_shapes(pred, gt)
    m = _valid_mask(gt, null_val)
    if not np.any(m):
        return float("nan")
    return float(np.abs(pred[m] - gt[m]).mean())


def masked_rmse(pred: Any, gt: Any, null_val: float = 0.0) -> float:
    """평균 제곱근 오차. 유효값 없으면 NaN."""
    np = _np()
    pred, gt = _check_shapes(pred, gt)
    m = _valid_mask(gt, null_val)
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(((pred[m] - gt[m]) ** 2).mean()))


def masked_mape(pred: Any, gt: Any, null_val: float = 0.0) -> float:
    """평균 절대 백분율 오차(%). |gt|<eps 위치는 추가로 제외(발산 방지)."""
    np = _np()
    pred, gt = _check_shapes(pred, gt)
    m = _valid_mask(gt, null_val)
    m &= np.abs(gt) > 1e-6
    if not np.any(m):
        return float("nan")
    return float((np.abs(pred[m] - gt[m]) / np.abs(gt[m])).mean() * 100.0)


# ---------------------------------------------------------------------
# 지평(horizon)별 지표
# ---------------------------------------------------------------------
def metrics_at_horizons(
    pred_seq: Any,
    gt_seq: Any,
    horizons: Iterable[int] = (3, 6, 12),
    null_val: float = 0.0,
) -> dict[str, Any]:
    """지평 3/6/12 스텝에서의 MAE/RMSE/MAPE.

    pred_seq / gt_seq: (T_out, *) — axis 0 이 예측 스텝. horizon h → 인덱스 h-1.
    반환: {"horizons": [...], "mae": {"h3":..}, "rmse": {...}, "mape": {...}}.
    범위를 벗어난 horizon 은 값을 만들지 않고 NaN.
    """
    np = _np()
    pred_seq, gt_seq = _check_shapes(pred_seq, gt_seq)
    if pred_seq.ndim < 1:
        raise ValueError("pred_seq/gt_seq 는 (T_out, *) 형상이어야 함")
    t_out = pred_seq.shape[0]
    out: dict[str, Any] = {"horizons": list(horizons), "mae": {}, "rmse": {}, "mape": {}}
    for h in horizons:
        key = f"h{h}"
        if h < 1 or h > t_out:
            out["mae"][key] = out["rmse"][key] = out["mape"][key] = float("nan")
            continue
        p, g = pred_seq[h - 1], gt_seq[h - 1]
        out["mae"][key] = masked_mae(p, g, null_val)
        out["rmse"][key] = masked_rmse(p, g, null_val)
        out["mape"][key] = masked_mape(p, g, null_val)
    return out


def metrics_per_step(
    pred_seq: Any,
    gt_seq: Any,
    null_val: float = 0.0,
) -> dict[str, Any]:
    """모든 예측 스텝별 MAE/RMSE/MAPE + 전체 평균.

    다단계 예측의 **오차 누적(발산)** 관찰용(RQ2). 반환:
      {"per_step": {"mae":[..T_out..], "rmse":[...], "mape":[...]}, "mean": {...}}.
    """
    np = _np()
    pred_seq, gt_seq = _check_shapes(pred_seq, gt_seq)
    t_out = pred_seq.shape[0]
    mae = [masked_mae(pred_seq[t], gt_seq[t], null_val) for t in range(t_out)]
    rmse = [masked_rmse(pred_seq[t], gt_seq[t], null_val) for t in range(t_out)]
    mape = [masked_mape(pred_seq[t], gt_seq[t], null_val) for t in range(t_out)]

    def _nanmean(xs):
        a = np.asarray(xs, dtype=np.float64)
        return float(np.nanmean(a)) if np.any(~np.isnan(a)) else float("nan")

    return {
        "per_step": {"mae": mae, "rmse": rmse, "mape": mape},
        "mean": {"mae": _nanmean(mae), "rmse": _nanmean(rmse), "mape": _nanmean(mape)},
    }
