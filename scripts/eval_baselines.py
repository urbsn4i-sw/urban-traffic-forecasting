#!/usr/bin/env python
"""eval_baselines.py — Phase 1: 실 METR-LA 기준선 실측.

파이프라인 (실데이터, 성능 보고용 — 합성 아님)
  1) metr-la.h5 (T, N) 적재
  2) 윈도우 12→12, 시간순 70/10/20 분할(셔플 없음)
  3) z-score 스케일러를 **train 구간만** 으로 fit → data/processed/scaler.json 저장(gitignore)
     (스케일러는 Phase 2 STGNN 용. 기준선/지표는 원 단위(mph)로 계산)
  4) 기준선(test split): copy-last(persistence) / seasonal Historical Average(DCRNN 정의)
  5) MAE/RMSE/MAPE @ horizon 3/6/12(=15/30/60분) + 스텝별(오차 누적, RQ2)
  6) results/<run_id>/{metrics.json, summary.md} 저장(실측값 + 실행 메타)

⚠️ 결측=0 은 마스크로 제외(masked 지표). 논문값은 참조용으로만 표기(subset/설정 차이 주의).
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve()
_TASK_DIR = _HERE.parents[1]
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_TASK_DIR / "src"))

from common.seeding import set_seed              # noqa: E402
from common.metrics import rollout_divergence    # noqa: E402
import metrics as M                               # noqa: E402
import baselines as B                             # noqa: E402
import data as D                                  # noqa: E402


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def load_config(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(series, t_in, horizon, ratios, period, null_val, horizons):
    """test split 에서 copy-last / seasonal-HA 를 실측. 반환 (results, meta)."""
    import numpy as np
    T, N = series.shape
    W = T - t_in - horizon + 1
    if W <= 0:
        raise ValueError("시계열이 너무 짧아 윈도우가 없습니다.")
    n_train, n_val, n_test = D.chronological_split_sizes(W, ratios)

    # test 윈도우 인덱스(전역) 와 타깃 시작 절대 인덱스
    test_w = np.arange(n_train + n_val, W)              # (n_test,)
    target_starts = test_w + t_in                       # 각 윈도우 첫 예측 스텝 절대 인덱스
    assert target_starts.max() + horizon - 1 < T

    # gt: (n_test, horizon, N)
    gt = np.stack([series[target_starts + h] for h in range(horizon)], axis=1)

    # copy-last: 마지막 관측(타깃 직전) 프레임을 전 지평 복사
    last_obs = series[target_starts - 1]                # (n_test, N)
    pred_copy = np.repeat(last_obs[:, None, :], horizon, axis=1)

    # seasonal HA: train 구간만으로 주기 평균표 → 타깃 시각 슬롯 조회
    table = B.seasonal_average_table(series, train_len=n_train + t_in, period=period, null_val=null_val)
    pred_ha = B.seasonal_ha_predict(table, target_starts, horizon, period)

    def _score(pred):
        # 지표는 (horizon, ...) 축이 0 이어야 하므로 horizon 을 앞으로.
        p = np.moveaxis(pred, 1, 0)                     # (horizon, n_test, N)
        g = np.moveaxis(gt, 1, 0)
        at_h = M.metrics_at_horizons(p, g, horizons=horizons, null_val=null_val)
        per_step = M.metrics_per_step(p, g, null_val=null_val)
        div = rollout_divergence(per_step["per_step"]["mae"])  # RQ2: 스텝별 MAE 발산
        return {"at_horizons": at_h, "per_step": per_step, "mae_divergence": div}

    results = {"copy_last": _score(pred_copy), "seasonal_historical_average": _score(pred_ha)}
    meta = {"num_windows": int(W), "n_train": int(n_train), "n_val": int(n_val),
            "n_test": int(n_test), "period": int(period)}
    return results, meta


DISPLAY = {"metr-la": "METR-LA", "pems-bay": "PEMS-BAY"}
# 논문 HA 참조값(reference only) — 데이터셋별. DCRNN(Li+ 2018) 논문 Table. 직접 비교 아님.
HA_REF = {
    "metr-la": {"mae": {"h3": 4.16, "h6": 4.16, "h12": 4.16},
                "rmse": {"h3": 7.80, "h6": 7.80, "h12": 7.80},
                "mape_pct": {"h3": 13.0, "h6": 13.0, "h12": 13.0}},
    "pems-bay": {"mae": {"h3": 2.88, "h6": 2.88, "h12": 2.88},
                 "rmse": {"h3": 5.59, "h6": 5.59, "h12": 5.59},
                 "mape_pct": {"h3": 6.8, "h6": 6.8, "h12": 6.8}},
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="교통 기준선 실측 (METR-LA / PEMS-BAY)")
    p.add_argument("--config", type=Path, default=_TASK_DIR / "config" / "base.yaml")
    p.add_argument("--dataset", default=None, choices=["metr-la", "pems-bay"],
                   help="config data.dataset 오버라이드")
    p.add_argument("--out", type=Path, default=None, help="결과 디렉토리(기본: results/<run_id>)")
    args = p.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    import numpy as np
    cfg = load_config(args.config)
    dataset = args.dataset or str(cfg["data"]["dataset"])
    ds = cfg["datasets"][dataset]
    ds_name = DISPLAY.get(dataset, dataset)
    seed = set_seed(int(cfg.get("seed", 42)))

    t_in = int(cfg["temporal"]["seq_len_in"])
    horizon = int(cfg["temporal"]["horizon"])
    horizons = tuple(cfg["metrics"]["horizons"])
    null_val = float(cfg["data"]["null_val"])
    ratios = (float(cfg["data"]["train_ratio"]), float(cfg["data"]["val_ratio"]),
              float(cfg["data"]["test_ratio"]))
    period = 7 * 24 * 12  # 1주 = 2016 스텝 @5분 (DCRNN HA 주기)

    data_dir = _TASK_DIR / cfg["data"]["root"]
    h5 = data_dir / ds["h5_file"]
    series = D.load_h5_traffic(h5)                     # (T, N); 없으면 FileNotFoundError
    T, N = series.shape

    # --- 스케일러(train 구간만) fit + 저장 (Phase 2 STGNN 용) ---
    W = T - t_in - horizon + 1
    n_train, n_val, n_test = D.chronological_split_sizes(W, ratios)
    scaler = D.Scaler.fit(series[: n_train + t_in], null_val=null_val)
    proc = data_dir / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "scaler.json").write_text(
        json.dumps({"mean": scaler.mean, "std": scaler.std, "null_val": null_val,
                    "fit_on": "train raw region only", "note": "z-score for Phase 2 STGNN"},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    # --- 기준선 실측 ---
    results, meta = evaluate(series, t_in, horizon, ratios, period, null_val, horizons)

    run_id = datetime.now(timezone.utc).strftime(f"baselines-{dataset}-%Y%m%dT%H%M%SZ")
    out_dir = args.out or (_TASK_DIR / "results" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "synthetic_dummy": False,
        "task": "urban-traffic-forecasting",
        "phase": "실 데이터 기준선",
        "dataset": {"name": ds_name, "id": dataset, "file": ds["h5_file"], "shape_T_N": [int(T), int(N)],
                    "sample_freq_min": 5, "null_val": null_val,
                    "missing_frac_pct": round(float((series == null_val).mean() * 100), 2),
                    "source": "liyaguang/DCRNN Google Drive (연구용 공개)"},
        "protocol": {"seq_len_in": t_in, "horizon": horizon, "horizons_steps": list(horizons),
                     "horizons_minutes": {f"h{h}": h * 5 for h in horizons},
                     "split_ratio": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
                     "split_sizes_windows": meta, "eval_split": "test",
                     "metric_masking": "null(=0) 제외 masked MAE/RMSE/MAPE(%)"},
        "run": {"seed": seed, "git_commit": _git_hash(),
                "timestamp_utc": run_id.split("-")[-1],
                "hardware": {"platform": platform.platform(), "processor": platform.processor() or "n/a",
                             "device": "CPU (기준선은 GPU 불필요)"},
                "scaler": {"mean": round(scaler.mean, 4), "std": round(scaler.std, 4),
                           "saved": "data/processed/scaler.json (gitignore)"}},
        "baselines_test": results,
        "reference_only": {
            "note": f"DCRNN 논문(Li+ 2018, Table 1) {ds_name} test HA 값. 구현 세부(HA 주기/결측처리) "
                    "차이로 직접 비교는 주의. 우리 결과로 옮겨 적지 않음.",
            "historical_average_paper": HA_REF.get(dataset, {}),
            "copy_last_paper": "논문 미보고(우리 기준선). 참조값 없음.",
        },
    }
    M.save_metrics(metrics, out_dir / "metrics.json")

    # --- summary.md ---
    def row(name, r):
        mae, rmse, mape = r["at_horizons"]["mae"], r["at_horizons"]["rmse"], r["at_horizons"]["mape"]
        d = r["mae_divergence"]
        return (f"| {name} | {mae['h3']:.3f} | {mae['h6']:.3f} | {mae['h12']:.3f} "
                f"| {rmse['h12']:.3f} | {mape['h12']:.2f} | {d['slope']:.4f} | {d['final_over_first']:.3f} |")
    lines = [
        f"# {ds_name} 기준선 실측 — {run_id}", "",
        f"- 데이터: {ds_name} (T,N)=({T},{N}), 결측 {metrics['dataset']['missing_frac_pct']}% (=0 마스크)",
        f"- 프로토콜: 과거 {t_in}스텝 → 미래 {horizon}스텝, 시간순 {ratios[0]:.0%}/{ratios[1]:.0%}/{ratios[2]:.0%}, eval=test",
        f"- 분할(윈도우): train {meta['n_train']} / val {meta['n_val']} / test {meta['n_test']} (총 {meta['num_windows']})",
        f"- seed={seed}, git={metrics['run']['git_commit'][:8]}, device=CPU", "",
        "## MAE/RMSE/MAPE (test, 원 단위 mph / %)", "",
        "| 모델 | MAE@15m | MAE@30m | MAE@60m | RMSE@60m | MAPE@60m(%) | MAE기울기(스텝당) | MAE@60/@5 |",
        "|---|---|---|---|---|---|---|---|",
        row("copy-last", results["copy_last"]),
        row("seasonal-HA", results["seasonal_historical_average"]),
        "",
        "> RQ2(오차 누적): copy-last 는 지평↑ 오차↑(기울기 양수), seasonal-HA 는 계절 슬롯 기반이라 "
        "지평에 거의 평탄(누적 없음) — 대조가 드러남.",
        f"> 참조용: 논문 HA MAE≈{HA_REF.get(dataset,{}).get('mae',{}).get('h12','—')}(평탄). 구현 차이로 직접 비교 주의(우리 결과 아님).",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- 콘솔 요약 ---
    print(f"[eval] {ds_name} 기준선 실측 완료 → {out_dir}")
    print(f"  data (T,N)=({T},{N}) 결측={metrics['dataset']['missing_frac_pct']}%  "
          f"windows: train {meta['n_train']}/val {meta['n_val']}/test {meta['n_test']}")
    for name, key in (("copy-last", "copy_last"), ("seasonal-HA", "seasonal_historical_average")):
        r = results[key]; mae = r["at_horizons"]["mae"]; d = r["mae_divergence"]
        print(f"  [{name}] MAE @15/30/60m = {mae['h3']:.3f} / {mae['h6']:.3f} / {mae['h12']:.3f}"
              f"   (누적 slope={d['slope']:.4f}, @60/@5={d['final_over_first']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
