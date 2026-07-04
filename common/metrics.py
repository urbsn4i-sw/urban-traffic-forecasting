"""공통 지표 유틸 — 롤아웃 오차 발산(누적) 정량화.

이 저장소(교통 STGNN)는 다단계 예측의 **오차 누적**(RQ2)을 관찰한다. 지평별 오차 수열이
얼마나 빠르게 커지는지를 아래 `rollout_divergence` 로 정량화한다.

설계 원칙
  - numpy 외 무거운 의존성 없음(지연 임포트). 미설치 환경에서도 모듈 import 는 되게.
  - 결과를 지어내지 않는다: 입력 수열에서 계산된 값만 반환. 유효점 <2 면 NaN.
"""
from __future__ import annotations

from typing import Any, Sequence


def _np():
    import numpy as np  # noqa: PLC0415
    return np


def rollout_divergence(error_seq: Sequence[float]) -> dict[str, Any]:
    """지평이 길어질수록 오차가 얼마나 커지는지(발산)를 정량화.

    입력: 지평별 '오차' 수열(예: 지평별 MAE). 클수록 나쁨.
    반환
      - slope:           오차 vs 지평 index 최소제곱 기울기(스텝당 증가량)
      - final_over_first: error[-1]/error[0] (첫 스텝 대비 마지막 배율)
      - monotonic_frac:  인접 스텝에서 오차가 증가한 비율(롤아웃 불안정 지표)
    NaN 은 계산에서 제외. 유효점 <2 면 해당 지표는 NaN.
    """
    np = _np()
    e = np.asarray(list(error_seq), dtype=np.float64)
    valid = ~np.isnan(e)
    ev = e[valid]

    if ev.size >= 2:
        x = np.arange(ev.size, dtype=np.float64)
        slope = float(np.polyfit(x, ev, 1)[0])
        diffs = np.diff(ev)
        monotonic_frac = float(np.count_nonzero(diffs > 0) / diffs.size)
        first = ev[0]
        final_over_first = float(ev[-1] / first) if first != 0 else float("inf")
    else:
        slope = float("nan")
        monotonic_frac = float("nan")
        final_over_first = float("nan")

    return {
        "slope": slope,
        "final_over_first": final_over_first,
        "monotonic_frac": monotonic_frac,
    }
