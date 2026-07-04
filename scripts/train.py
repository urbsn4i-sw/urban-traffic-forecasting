#!/usr/bin/env python
"""train.py — 교통 STGNN 학습 진입점 **골격**.

⚠️ Phase 0 상태: **골격**. 실 데이터 학습은 Phase 1 에서 붙인다. 지금은
  - `--dry-run`: 실 데이터 없이 **합성 배치 1개로 모델 wiring(순전파/역전파 1스텝)만 검증**.
    (torch 필요. 없으면 안내 후 종료. 성능 보고 아님.)
  - 데이터 인자가 주어졌는데 파일이 없으면 값을 지어내지 않고 명확히 실패한다.

재현성: 시작부에서 common.seeding.set_seed(cfg.seed). 하이퍼파라미터는 config/*.yaml.
실행 메타(run_id/config 스냅샷/git hash/하드웨어/지표)는 Phase 1 에서 results/<run_id>/ 에 기록한다.
"""
from __future__ import annotations

import os
# anaconda(MKL libiomp5) + pip torch 의 OpenMP 런타임 중복 로드 충돌 회피(Windows).
# GPU-vs-CPU 수치 일치를 확인(diff<1e-3)했으므로 결과 정확성에는 영향 없음.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_TASK_DIR = _HERE.parents[1]
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))          # common.*
sys.path.insert(0, str(_TASK_DIR / "src"))   # model / data / graph / metrics

from common.seeding import set_seed          # noqa: E402


def load_config(path: Path) -> dict:
    import yaml  # noqa: PLC0415
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dry_run(cfg: dict) -> int:
    """실 데이터 없이 합성 배치로 모델 순전파/역전파 1스텝을 검증(wiring 확인).

    **모든 인접행렬 모드**(fixed/learned/hybrid/identity)를 각각 build→forward→shape assert→backward.
    특히 hybrid(2 supports)는 이전 GraphConv 차원 버그가 있던 경로 — 형상 assert 로 실측 검증한다.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        print("[train:dry-run] torch 미설치 — requirements.txt 스택 설치 후 재실행.")
        return 0

    import numpy as np  # noqa: PLC0415
    import model as Model  # noqa: PLC0415
    import graph as G      # noqa: PLC0415

    seed = set_seed(int(cfg.get("seed", 42)))
    n = 8
    t_in = int(cfg["temporal"]["seq_len_in"])
    horizon = int(cfg["temporal"]["horizon"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 합성 고정 인접행렬(거리 → 가우시안 커널 → 랜덤워크 정규화)
    rng = np.random.default_rng(0)
    dist = rng.uniform(0, 10, size=(n, n)); dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)
    adj = G.normalize_adj_random_walk(G.gaussian_kernel_adjacency(dist))

    print(f"[train:dry-run] device={device} torch={torch.__version__} seed={seed} "
          f"t_in={t_in} horizon={horizon} nodes={n}")
    all_ok = True
    for adj_mode in Model.ADJ_MODES:
        net = Model.build_model(
            num_nodes=n, in_dim=1, out_dim=1, horizon=horizon,
            hidden=int(cfg["model"]["hidden"]), n_layers=int(cfg["model"]["n_layers"]),
            diffusion_order=int(cfg["model"]["diffusion_order"]),
            adj_mode=adj_mode, adj_fixed=adj,
        ).to(device)
        x = torch.randn(4, t_in, n, 1, device=device)
        y = torch.randn(4, horizon, n, 1, device=device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        net.train()
        try:
            out = net(x)
            assert out.shape == y.shape, f"출력 형상 불일치: {out.shape} vs {y.shape}"
            loss = torch.nn.functional.l1_loss(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            n_params = sum(p.numel() for p in net.parameters())
            print(f"  [OK] adj_mode={adj_mode:8s} out={tuple(out.shape)} "
                  f"params={n_params} loss1={loss.item():.4f}")
        except Exception as e:  # noqa: BLE001
            all_ok = False
            print(f"  [FAIL] adj_mode={adj_mode:8s} → {type(e).__name__}: {e}")
    print("[train:dry-run] 결과:", "ALL PASS (합성; 성능 아님)" if all_ok else "일부 FAIL")
    return 0 if all_ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="교통 STGNN 학습(골격)")
    p.add_argument("--config", type=Path, default=_TASK_DIR / "config" / "base.yaml")
    p.add_argument("--dry-run", action="store_true",
                   help="실 데이터 없이 합성 배치로 모델 wiring 만 검증")
    args = p.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    cfg = load_config(args.config)

    if args.dry_run:
        return dry_run(cfg)

    # --- 실 학습 경로: Phase 1 에서 구현 ---
    print("[train] 실 데이터 학습은 Phase 1 에서 구현됩니다. 지금은 --dry-run 만 지원합니다.")
    print("        (데이터: scripts/download_data.sh 로 취득 → data.py 로 윈도우 → 학습 루프)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
