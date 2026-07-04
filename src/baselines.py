"""교통 STGNN 단순 기준선 — 파이프라인 검증 + 비교 기준.

기준선 정의 (DCRNN / Graph WaveNet 논문 관례)
  - **copy-last (persistence):** 마지막 관측 프레임을 미래 전 지평에 복사. 교통은 자기상관이
    높아 단기(15분)에서 의외로 강한 기준선.
  - **Historical Average (HA):** 관측 history 평균을 미래 전 지평에 사용(단순형). 논문의 HA 는
    보통 '요일×시간대' 계절 평균을 쓰지만, 여기서는 학습/스모크 단계용 단순형을 제공하고
    계절형(seasonal)은 Phase 1 에서 데이터가 있을 때 확장한다.

설계 원칙: numpy 만 사용, 지연 임포트, 값 지어내지 않음(입력에서 계산된 값만).
형상 규약: history (T_in, N[, C]), 반환 (horizon, N[, C]).
"""
from __future__ import annotations

from typing import Any


def _np():
    import numpy as np  # noqa: PLC0415
    return np


def copy_last(history: Any, horizon: int) -> Any:
    """마지막 관측 프레임을 horizon 만큼 복사. 반환 (horizon, *history.shape[1:])."""
    np = _np()
    hist = np.asarray(history, dtype=np.float64)
    if hist.ndim < 1 or hist.shape[0] < 1:
        raise ValueError("history 는 (T_in, ...) 이고 T_in>=1 이어야 함")
    last = hist[-1]
    return np.repeat(last[None, ...], horizon, axis=0)


def seasonal_average_table(series: Any, train_len: int, period: int, null_val: float = 0.0):
    """주기(period) 슬롯별 계절 평균표 (period, N) — **train 구간만** 사용(누수 방지).

    DCRNN 논문의 Historical Average 정의: 각 시각을 '주(week) 주기 내 같은 슬롯'의
    과거 평균으로 예측(period=7일=2016스텝@5분). null_val 위치는 평균에서 제외.
    어떤 슬롯/노드에 유효 관측이 없으면 NaN(값을 지어내지 않음).
    """
    np = _np()
    s = np.asarray(series, dtype=np.float64)[:train_len]
    if s.ndim != 2:
        raise ValueError(f"series 는 (T, N) 이어야 함: {s.shape}")
    T, N = s.shape
    mask = (~np.isnan(s)) if (isinstance(null_val, float) and np.isnan(null_val)) else (s != null_val)
    table_sum = np.zeros((period, N))
    table_cnt = np.zeros((period, N))
    slots = np.arange(T) % period
    np.add.at(table_sum, slots, np.where(mask, s, 0.0))
    np.add.at(table_cnt, slots, mask.astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        table = np.where(table_cnt > 0, table_sum / table_cnt, np.nan)
    return table


def seasonal_ha_predict(table: Any, target_start_indices: Any, horizon: int, period: int) -> Any:
    """계절 평균표로 다중 윈도우 예측. 반환 (num_windows, horizon, N).

    target_start_indices[w]: 윈도우 w 의 첫 예측 타임스텝 절대 인덱스.
    horizon h(0-based) 의 절대 인덱스 = target_start + h → 슬롯 (…)%period 조회.
    HA 는 타깃 시각의 계절 슬롯만 보므로 지평에 따라 예측이 '누적 변화'하지 않는다(대조).
    """
    np = _np()
    tbl = np.asarray(table, dtype=np.float64)
    starts = np.asarray(target_start_indices, dtype=np.int64)[:, None]  # (W,1)
    h = np.arange(horizon)[None, :]                                     # (1,H)
    slots = (starts + h) % period                                       # (W,H)
    return tbl[slots]                                                    # (W,H,N)


def historical_average(history: Any, horizon: int, null_val: float = 0.0) -> Any:
    """관측 history 의 (결측 제외) 평균을 미래 전 지평에 사용. 반환 (horizon, *).

    null_val 위치는 평균에서 제외한다(교통 결측=0 관례). 어떤 노드에서 유효 관측이
    하나도 없으면 그 노드는 NaN(값을 지어내지 않음).
    """
    np = _np()
    hist = np.asarray(history, dtype=np.float64)
    if hist.ndim < 1 or hist.shape[0] < 1:
        raise ValueError("history 는 (T_in, ...) 이고 T_in>=1 이어야 함")
    if null_val is None or (isinstance(null_val, float) and np.isnan(null_val)):
        mask = ~np.isnan(hist) if null_val is not None else np.ones_like(hist, dtype=bool)
    else:
        mask = hist != null_val
    summed = np.where(mask, hist, 0.0).sum(axis=0)
    counts = mask.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, summed / counts, np.nan)
    return np.repeat(mean[None, ...], horizon, axis=0)
