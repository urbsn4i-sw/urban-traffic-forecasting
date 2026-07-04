"""교통 STGNN 데이터 유틸 — 윈도우 생성 · z-score 정규화 · 합성 생성기.

Phase 0 범위: **로직 골격만**. 실 METR-LA/PEMS-BAY(.h5) 적재는 Phase 1 에서 붙인다.
  - `make_windows` / `Scaler` / `synthetic_traffic` 는 numpy 만으로 동작(스모크·CI 대상).
  - `load_h5_traffic` 는 pandas 지연 임포트(Phase 1 데이터 준비 후 사용). 데이터가 없으면
    값을 지어내지 않고 명확한 에러를 낸다.

형상 규약: 시계열 배열은 (T, N) — T=타임스텝, N=센서(노드). 5분 간격.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def _np():
    import numpy as np  # noqa: PLC0415
    return np


class Scaler:
    """z-score 표준화. **학습 split 통계로만 fit** 하고 val/test 에 적용(누수 방지)."""

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = float(mean)
        self.std = float(std) if std != 0 else 1.0

    @classmethod
    def fit(cls, arr: Any, null_val: float = 0.0) -> "Scaler":
        np = _np()
        a = np.asarray(arr, dtype=np.float64)
        m = a != null_val if not np.isnan(null_val) else ~np.isnan(a)
        vals = a[m]
        if vals.size == 0:
            raise ValueError("Scaler.fit: 유효 관측이 없습니다(값을 지어내지 않음).")
        return cls(mean=float(vals.mean()), std=float(vals.std()))

    def transform(self, arr: Any) -> Any:
        np = _np()
        return (np.asarray(arr, dtype=np.float64) - self.mean) / self.std

    def inverse_transform(self, arr: Any) -> Any:
        np = _np()
        return np.asarray(arr, dtype=np.float64) * self.std + self.mean


def chronological_split_sizes(num_windows: int, ratios=(0.7, 0.1, 0.2)):
    """윈도우 개수를 시간순 train/val/test 로 분할한 크기 (n_train, n_val, n_test).

    DCRNN 관례: n_test=round(W*test), n_train=round(W*train), n_val=나머지(누수 없이 합=W).
    분할은 **셔플 없이 시간순**(앞=train, 중간=val, 뒤=test).
    """
    tr, va, te = ratios
    n_test = int(round(num_windows * te))
    n_train = int(round(num_windows * tr))
    n_val = num_windows - n_train - n_test
    if n_val < 0:
        raise ValueError(f"분할 비율이 잘못됨: {ratios} → n_val={n_val}")
    return n_train, n_val, n_test


def make_windows(series: Any, t_in: int, horizon: int):
    """(T, N) 시계열 → 슬라이딩 윈도우 (X, Y).

    X: (num_windows, t_in, N)   과거 관측
    Y: (num_windows, horizon, N) 미래 타깃
    윈도우가 안 나오면 빈 배열(0-length)을 반환(값을 지어내지 않음).
    """
    np = _np()
    s = np.asarray(series, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f"series 는 (T, N) 이어야 함: {s.shape}")
    T = s.shape[0]
    n = T - t_in - horizon + 1
    if n <= 0:
        empty_x = np.empty((0, t_in, s.shape[1]))
        empty_y = np.empty((0, horizon, s.shape[1]))
        return empty_x, empty_y
    X = np.stack([s[i : i + t_in] for i in range(n)], axis=0)
    Y = np.stack([s[i + t_in : i + t_in + horizon] for i in range(n)], axis=0)
    return X, Y


def synthetic_traffic(
    num_steps: int = 288 * 3,
    num_nodes: int = 8,
    seed: int = 42,
    period: int = 288,
):
    """합성 교통 속도 시계열 (T, N) — 일 주기(period=하루=288스텝@5분) + 노드별 위상 + 잡음.

    스모크/CI 전용. 성능 보고가 아니며, 지평이 길수록 예측이 어려워지도록 잡음을 설계.
    """
    np = _np()
    rng = np.random.default_rng(seed)
    t = np.arange(num_steps)[:, None]                       # (T,1)
    phase = rng.uniform(0, 2 * np.pi, size=(1, num_nodes))   # 노드별 위상
    base = 60.0 + 15.0 * np.sin(2 * np.pi * t / period + phase)  # 일 주기 속도(마일/h 스케일)
    noise = rng.normal(0, 3.0, size=(num_steps, num_nodes))
    speed = np.clip(base + noise, 0.0, None)
    return speed


def build_distance_matrix(dist_csv: str | Path, sensor_ids, has_header: bool = True):
    """센서 거리 CSV(from,to,cost) → 거리 행렬 (N,N), sensor_ids 순서로 정렬.

    미연결 쌍은 inf, 자기 자신은 0. 이후 graph.gaussian_kernel_adjacency 로 가중치화.
    (DCRNN 방식: 이 거리행렬의 유한값 std 를 sigma 로 사용.)
    has_header: 첫 줄이 헤더(from,to,cost)면 True(METR-LA), 헤더 없으면 False(PEMS-BAY).
    """
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    ids = [str(s) for s in sensor_ids]
    pos = {sid: i for i, sid in enumerate(ids)}
    n = len(ids)
    D = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(D, 0.0)
    if has_header:
        df = pd.read_csv(dist_csv, dtype={"from": str, "to": str})
    else:
        df = pd.read_csv(dist_csv, header=None, names=["from", "to", "cost"],
                         dtype={"from": str, "to": str})
    for frm, to, cost in df.itertuples(index=False):
        if frm in pos and to in pos:
            D[pos[frm], pos[to]] = cost
    return D


def load_h5_traffic(h5_path: str | Path, key: Optional[str] = None):
    """실 METR-LA/PEMS-BAY .h5 → (T, N) numpy. pandas 지연 임포트.

    Phase 1 용. 파일이 없으면 값을 지어내지 않고 FileNotFoundError.
    """
    p = Path(h5_path)
    if not p.exists():
        raise FileNotFoundError(
            f"교통 데이터 파일이 없습니다: {p}\n"
            "→ scripts/download_data.sh 안내에 따라 취득·배치하세요(대용량, git 미포함)."
        )
    import pandas as pd  # noqa: PLC0415  (지연 임포트: Phase 0 미설치 허용)
    df = pd.read_hdf(p, key=key) if key else pd.read_hdf(p)
    return df.values
