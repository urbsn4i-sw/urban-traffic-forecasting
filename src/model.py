"""교통 STGNN 소형 모델 골격 — Graph WaveNet-lite.

⚠️ **Phase 0 상태: 골격(skeleton)**. import 가능하고 forward 형상은 맞지만, 아직
   학습·검증하지 않았다(성능 미보고). 실제 학습/평가는 Phase 1(train.py / eval.py).

설계 (브리지 논문 대응)
  - **공간 확산 그래프 합성곱**(DCRNN diffusion conv 축소형): 고정 전이행렬 A 의 K-차
    거듭제곱 합 Σ_k A^k X W_k.
  - **학습(adaptive) 인접행렬**(Graph WaveNet): 노드 임베딩 E1, E2 → softmax(relu(E1 E2^T)).
  - **시간 합성곱**: 게이트드 causal Conv1d(시간축). (원 논문의 dilated stack 을 축소.)

절제 실험 1(RQ1)을 위한 인접행렬 모드(`adj_mode`)
  - "fixed"    : 고정 A 만 사용(도로망 거리 기반, graph.py).
  - "learned"  : adaptive A 만 사용(A_fixed 무시).
  - "hybrid"   : 고정 + adaptive 둘 다(supports 합).
  - "identity" : 그래프 없음(각 노드 독립) — 대조군.

torch 는 **지연 임포트**(Phase 0 미설치 환경에서 이 파일 import 자체는 안 깨지도록,
실제 클래스 정의는 build_model() 호출 시점에 이뤄진다).
"""
from __future__ import annotations

from typing import Any, Optional

ADJ_MODES = ("fixed", "learned", "hybrid", "identity")


def build_model(
    num_nodes: int,
    in_dim: int = 1,
    out_dim: int = 1,
    horizon: int = 12,
    hidden: int = 32,
    n_layers: int = 2,
    diffusion_order: int = 2,
    adj_mode: str = "hybrid",
    adj_fixed: Any = None,
    node_emb_dim: int = 10,
    dropout: float = 0.3,
):
    """STGNN 인스턴스를 생성해 반환. torch 는 이 함수 안에서 임포트한다.

    adj_fixed: (N, N) 정규화된 고정 전이행렬(numpy/tensor). adj_mode 가 fixed/hybrid 면 필요.
    """
    if adj_mode not in ADJ_MODES:
        raise ValueError(f"adj_mode 는 {ADJ_MODES} 중 하나여야 함: {adj_mode!r}")

    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    class GraphConv(nn.Module):
        """확산 그래프 합성곱: supports(전이행렬 리스트)의 K-차 거듭제곱 합.

        출력 채널 수 = c_in * (order * num_supports + 1)  [자기항 1 + support별 K차].
        선형층을 이 크기에 맞춰야 하므로 num_supports 를 명시로 받는다.
        """

        def __init__(self, c_in, c_out, order, num_supports):
            super().__init__()
            self.order = order
            self.lin = nn.Linear(c_in * (order * num_supports + 1), c_out)

        def forward(self, x, supports):
            # x: (B, N, C).  supports: [(N, N), ...] 정규화 전이행렬.
            out = [x]
            for adj in supports:
                h = x
                for _ in range(self.order):
                    h = torch.einsum("nm,bmc->bnc", adj, h)
                    out.append(h)
            h = torch.cat(out, dim=-1)          # (B, N, C*(order*|supports|+1))
            return self.lin(h)

    class STBlock(nn.Module):
        """게이트드 시간 conv → 그래프 conv → 잔차."""

        def __init__(self, c, order, num_supports, kernel=2, dilation=1):
            super().__init__()
            pad = (kernel - 1) * dilation
            self.pad = pad
            self.filt = nn.Conv1d(c, c, kernel, dilation=dilation)
            self.gate = nn.Conv1d(c, c, kernel, dilation=dilation)
            # GraphConv 입력 차원은 support 개수에 의존 → num_supports 를 전달
            self.gconv = GraphConv(c, c, order, num_supports)
            self.num_supports = num_supports
            self.norm = nn.LayerNorm(c)

        def forward(self, x, supports):
            # x: (B, C, N, T)
            b, c, n, t = x.shape
            xt = x.permute(0, 2, 1, 3).reshape(b * n, c, t)     # (B*N, C, T)
            xt = F.pad(xt, (self.pad, 0))                        # causal
            tc = torch.tanh(self.filt(xt)) * torch.sigmoid(self.gate(xt))
            tc = tc.reshape(b, n, c, -1).permute(0, 2, 1, 3)     # (B, C, N, T)
            # 그래프 합성곱은 마지막 타임스텝 특징에 적용(축소형)
            last = tc[..., -1].permute(0, 2, 1)                  # (B, N, C)
            g = self.gconv(last, supports)                       # (B, N, C)
            g = self.norm(g)
            return tc, g

    class STGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.adj_mode = adj_mode
            self.horizon = horizon
            self.out_dim = out_dim
            self.num_nodes = num_nodes
            self.dropout = dropout

            self.input_proj = nn.Linear(in_dim, hidden)

            # 고정 전이행렬(버퍼) — 학습되지 않음
            if adj_fixed is not None:
                a = torch.as_tensor(adj_fixed, dtype=torch.float32)
                self.register_buffer("adj_fixed", a)
            else:
                self.register_buffer("adj_fixed", torch.eye(num_nodes))

            # 학습(adaptive) 인접행렬용 노드 임베딩
            if adj_mode in ("learned", "hybrid"):
                self.e1 = nn.Parameter(torch.randn(num_nodes, node_emb_dim) * 0.05)
                self.e2 = nn.Parameter(torch.randn(num_nodes, node_emb_dim) * 0.05)
            else:
                self.e1 = self.e2 = None

            num_supports = 1 if adj_mode != "hybrid" else 2
            self.blocks = nn.ModuleList(
                [STBlock(hidden, diffusion_order, num_supports, dilation=2 ** i)
                 for i in range(n_layers)]
            )
            self.head = nn.Linear(hidden, horizon * out_dim)

        def _supports(self):
            import torch.nn.functional as F  # noqa: PLC0415
            if self.adj_mode == "identity":
                return [torch.eye(self.num_nodes, device=self.adj_fixed.device)]
            if self.adj_mode == "fixed":
                return [self.adj_fixed]
            adp = F.softmax(F.relu(self.e1 @ self.e2.t()), dim=1)
            if self.adj_mode == "learned":
                return [adp]
            return [self.adj_fixed, adp]  # hybrid

        def forward(self, x):
            # x: (B, T_in, N, in_dim)
            import torch  # noqa: PLC0415
            import torch.nn.functional as F  # noqa: PLC0415
            b, t, n, _ = x.shape
            h = self.input_proj(x)                      # (B, T, N, hidden)
            h = h.permute(0, 3, 2, 1)                   # (B, hidden, N, T)
            supports = self._supports()
            g_last = None
            for blk in self.blocks:
                h, g_last = blk(h, supports)
                h = F.dropout(h, self.dropout, self.training)
            out = self.head(g_last)                     # (B, N, horizon*out_dim)
            out = out.reshape(b, n, self.horizon, self.out_dim)
            return out.permute(0, 2, 1, 3)              # (B, horizon, N, out_dim)

    return STGNN()
