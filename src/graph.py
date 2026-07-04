"""교통 STGNN 그래프(인접행렬) 구성 — **고정(static) 인접행렬** 유틸.

절제 실험 1 (RQ1: 인접행렬 고정 vs 학습)에서 '고정' 쪽에 해당하는 부분.
  - **고정 인접행렬**은 센서 간 도로망 거리로부터 만든다(DCRNN 방식): 임계 가우시안 커널.
  - **학습(adaptive) 인접행렬**은 torch 파라미터가 필요하므로 model.py(Graph WaveNet 방식,
    노드 임베딩 E1·E2 → softmax(relu(E1 E2^T)))에서 다룬다.

설계 원칙: numpy 만 사용, 지연 임포트, 값 지어내지 않음.
"""
from __future__ import annotations

from typing import Any, Optional


def _np():
    import numpy as np  # noqa: PLC0415
    return np


def gaussian_kernel_adjacency(
    dist_mx: Any,
    sigma: Optional[float] = None,
    threshold: float = 0.1,
) -> Any:
    """센서 거리 행렬 → 가중 인접행렬 W_ij = exp(-(d_ij/sigma)^2) (DCRNN eq.).

    - sigma 기본값: 유한 거리들의 표준편차(논문 관례).
    - threshold 미만 가중치는 0 으로 만들어 희소화(자기루프 포함 대각은 보존).
    - dist_mx 의 inf/NaN(도달 불가)은 가중치 0 으로 처리.
    반환: (N, N) float64 밀집 행렬.
    """
    np = _np()
    d = np.asarray(dist_mx, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError(f"dist_mx 는 정방 (N,N) 이어야 함: {d.shape}")
    finite = d[np.isfinite(d)]
    if sigma is None:
        sigma = float(finite.std()) if finite.size else 1.0
    if sigma == 0:
        sigma = 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        w = np.exp(-((d / sigma) ** 2))
    w[~np.isfinite(d)] = 0.0
    w[w < threshold] = 0.0
    return w


def normalize_adj_random_walk(adj: Any) -> Any:
    """랜덤워크 정규화 D^{-1} A (DCRNN diffusion conv 의 전이행렬). 고립 노드는 0 행."""
    np = _np()
    a = np.asarray(adj, dtype=np.float64)
    deg = a.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_inv = np.where(deg > 0, 1.0 / deg, 0.0)
    return d_inv[:, None] * a


def normalize_adj_symmetric(adj: Any) -> Any:
    """대칭 정규화 D^{-1/2} (A+I) D^{-1/2} (GCN 방식). 자기루프 추가."""
    np = _np()
    a = np.asarray(adj, dtype=np.float64)
    a = a + np.eye(a.shape[0])
    deg = a.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0)
    return d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]


def identity_adjacency(num_nodes: int) -> Any:
    """단위행렬 = '그래프 없음' 절제(각 노드가 자기 자신만 참조). RQ1 대조군."""
    np = _np()
    return np.eye(int(num_nodes), dtype=np.float64)
