#!/usr/bin/env python
"""smoke 용 더미 파이프라인: 합성 교통 시계열 → 기준선 → 지표가 끝까지 도는지 검증.

⚠️ 이 스크립트는 **합성(dummy) 텐서**만 쓴다. 실 METR-LA/PEMS-BAY 데이터도, STGNN 모델(torch)도
   필요/사용하지 않는다(numpy 만). 출력 수치는 파이프라인 동작 확인용이며 **성능 보고가 아니다**
   (metrics.json 의 synthetic_dummy=true 로 표시).

하는 일
  1) set_seed 로 시드 고정
  2) 합성 교통 속도 시계열 (T, N) 생성(일 주기 + 노드 위상 + 잡음)
  3) 윈도우화 → copy-last / Historical Average 기준선 예측
  4) horizon 3/6/12(=15/30/60분) MAE/RMSE/MAPE + 스텝별(오차 누적) + 발산 지표
  5) metrics.json 저장(--out) + 요약 출력
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_TASK_DIR = _HERE.parents[1]
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))          # common.*
sys.path.insert(0, str(_TASK_DIR / "src"))   # metrics / baselines / data

from common.seeding import set_seed          # noqa: E402
from common.metrics import rollout_divergence  # noqa: E402  (오차 누적 정량화 재사용)
import metrics as M                           # noqa: E402  (traffic 지표)
import baselines as B                         # noqa: E402
import data as D                              # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="교통 기준선→지표 더미 파이프라인(smoke)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--t-in", type=int, default=12)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--num-nodes", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=288 * 3)
    p.add_argument("--out", type=Path, required=True, help="metrics.json 저장 디렉토리")
    args = p.parse_args(argv)

    # Windows 콘솔(cp949)에서도 한글/기호 출력이 안 깨지게 UTF-8 강제.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    import numpy as np
    seed = set_seed(args.seed)

    # --- 합성 시계열 → 윈도우 ---
    series = D.synthetic_traffic(args.num_steps, args.num_nodes, seed=seed)
    X, Y = D.make_windows(series, args.t_in, args.horizon)  # X:(W,t_in,N) Y:(W,horizon,N)
    if X.shape[0] == 0:
        print("[smoke] FAIL: 윈도우가 생성되지 않음(시계열이 너무 짧음).", file=sys.stderr)
        return 1

    # --- 기준선 예측 (윈도우별) → (W, horizon, N) ---
    pred_copy = np.stack([B.copy_last(x, args.horizon) for x in X], axis=0)
    pred_ha = np.stack([B.historical_average(x, args.horizon) for x in X], axis=0)

    # 지표는 (horizon, ...) 축이 0 이어야 하므로 (horizon, W, N) 으로 옮겨 스텝축을 앞으로.
    def _hz_first(a):
        return np.moveaxis(a, 1, 0)  # (horizon, W, N)

    gt = _hz_first(Y)
    results = {}
    for name, pred in (("copy_last", pred_copy), ("historical_average", pred_ha)):
        ph = _hz_first(pred)
        at_h = M.metrics_at_horizons(ph, gt, horizons=(3, 6, 12), null_val=float("nan"))
        per_step = M.metrics_per_step(ph, gt, null_val=float("nan"))
        div = rollout_divergence(per_step["per_step"]["mae"])  # 스텝별 MAE 발산
        results[name] = {"at_horizons": at_h, "per_step": per_step, "mae_divergence": div}

    metrics = {
        "synthetic_dummy": True,
        "note": "합성 교통 시계열 기반 파이프라인 검증. 성능 보고 아님(실데이터/모델 미사용).",
        "config": {
            "seed": seed, "t_in": args.t_in, "horizon": args.horizon,
            "num_nodes": args.num_nodes, "num_steps": args.num_steps,
            "sample_freq_min": 5, "horizon_minutes": {"h3": 15, "h6": 30, "h12": 60},
        },
        "baselines": results,
    }
    out_path = M.save_metrics(metrics, Path(args.out) / "metrics.json")

    # --- 요약 출력 ---
    print("[SMOKE/DUMMY] 합성 교통 시계열 파이프라인 — 성능 보고 아님")
    print(f"  seed={seed} t_in={args.t_in} horizon={args.horizon} nodes={args.num_nodes} windows={X.shape[0]}")
    for name in ("copy_last", "historical_average"):
        mae = results[name]["at_horizons"]["mae"]
        div = results[name]["mae_divergence"]
        print(f"  [{name}] MAE @15/30/60m = "
              f"{mae['h3']:.3f} / {mae['h6']:.3f} / {mae['h12']:.3f}"
              f"   (오차 누적 slope={div['slope']:.4f}, final/first={div['final_over_first']:.2f})")
    print(f"  → metrics.json 저장: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
