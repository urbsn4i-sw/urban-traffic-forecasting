#!/usr/bin/env python
"""eval.py — 교통 STGNN 평가 진입점 **골격**.

⚠️ Phase 0 상태: **골격**. 실 데이터·체크포인트 평가는 Phase 1 에서 붙인다.
  - 지표 계산 로직 자체는 구현되어 있다: src/metrics.py 의
    metrics_at_horizons(3/6/12) · metrics_per_step(오차 누적).
  - 데이터/ckpt 가 없으면 값을 지어내지 않고 명확히 실패한다.

산출: results/<run_id>/metrics.json (Phase 1). 지금은 경로/계약만 확정한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_TASK_DIR = _HERE.parents[1]
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_TASK_DIR / "src"))

from common.seeding import set_seed          # noqa: E402


def load_config(path: Path) -> dict:
    import yaml  # noqa: PLC0415
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="교통 STGNN 평가(골격)")
    p.add_argument("--config", type=Path, default=_TASK_DIR / "config" / "base.yaml")
    p.add_argument("--ckpt", type=Path, default=None, help="학습된 체크포인트(.pt) 경로")
    args = p.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    if args.ckpt is None or not args.ckpt.exists():
        print("[eval] 체크포인트가 없습니다. 실 평가는 Phase 1(학습 후) 에서 진행합니다.")
        print("       지표 계약: metrics_at_horizons(h=3/6/12) + metrics_per_step(오차 누적)")
        print("       → 결과는 results/<run_id>/metrics.json 으로 저장(git 허용 산출물).")
        return 0

    # --- 실 평가 경로: Phase 1 에서 구현 ---
    print("[eval] 실 데이터·모델 평가는 Phase 1 에서 구현됩니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
