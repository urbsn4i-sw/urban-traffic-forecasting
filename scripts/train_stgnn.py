#!/usr/bin/env python
"""train_stgnn.py — Phase 2-B: 소형 STGNN 실제 학습 + 인접행렬 절제(RQ1).

파이프라인 (실 METR-LA, 성능 보고용)
  1) metr-la.h5 (T,N) 적재 → 특징 [z-score(speed), time-of-day] (F=2)
  2) 윈도우 12→12, 시간순 70/10/20 (Phase 1 과 동일 분할)
  3) 인접행렬(고정): 센서 거리 CSV → 가우시안 커널 → random-walk 정규화
  4) STGNN 을 4개 인접행렬 모드(fixed/learned/hybrid/identity)로 **같은 조건** 학습(공정 비교)
  5) test split 에서 MAE/RMSE/MAPE @3/6/12 + 오차 누적 → 기준선(copy-last·seasonal-HA)과 합산
  6) results/<run_id>/{metrics.json, summary.md} (실측값 + 실행 메타)

⚠️ SOTA 재현 아님(원리 재현). 결측=0 마스크. 논문값은 참조용 분리. 값 지어내지 않음.
가중치(*.pt)는 선택 저장 시에도 .gitignore 로 커밋 차단.
"""
from __future__ import annotations

import os
# anaconda(MKL) + torch OpenMP DLL 충돌 회피(Windows). GPU-vs-CPU 수치 일치 확인 → 정확성 무해.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import platform
import subprocess
import sys
import time
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
import graph as G                                 # noqa: E402
import model as Model                             # noqa: E402


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT),
                                       text=True).strip()
    except Exception:
        return "unknown"


def load_config(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare(cfg, device):
    """특징/타깃 텐서·분할 인덱스·고정 인접행렬 준비. 모든 텐서를 device 에 올린다."""
    import numpy as np
    import pandas as pd
    import torch

    t_in = int(cfg["temporal"]["seq_len_in"])
    horizon = int(cfg["temporal"]["horizon"])
    null_val = float(cfg["data"]["null_val"])
    ratios = (float(cfg["data"]["train_ratio"]), float(cfg["data"]["val_ratio"]),
              float(cfg["data"]["test_ratio"]))
    data_dir = _TASK_DIR / cfg["data"]["root"]

    df = pd.read_hdf(data_dir / cfg["data"]["h5_file"])
    speed = df.values.astype(np.float64)                      # (T, N) 원 단위(mph)
    T, N = speed.shape
    idx = pd.DatetimeIndex(df.index)
    tod = ((idx.hour * 3600 + idx.minute * 60 + idx.second) / 86400.0).to_numpy()  # (T,)

    W = T - t_in - horizon + 1
    n_train, n_val, n_test = D.chronological_split_sizes(W, ratios)

    # 스케일러: train 구간(입력 타임스텝)만으로 fit (누수 방지)
    scaler = D.Scaler.fit(speed[: n_train + t_in], null_val=null_val)
    speed_z = scaler.transform(speed)                          # (T,N)
    feat = np.stack([speed_z, np.repeat(tod[:, None], N, axis=1)], axis=-1)  # (T,N,2)

    # 고정 인접행렬: 거리 CSV → 커널 → random-walk 정규화 (sensor_ids = h5 컬럼 순서)
    sensor_ids = [str(c) for c in df.columns]
    Dmx = D.build_distance_matrix(data_dir / "sensor_graph" / "distances_la_2012.csv", sensor_ids)
    adj = G.normalize_adj_random_walk(
        G.gaussian_kernel_adjacency(Dmx, threshold=float(cfg["model"]["adjacency"]["kernel_threshold"])))
    edge_density = float((adj > 0).mean())

    tens = {
        "feat": torch.tensor(feat, dtype=torch.float32, device=device),      # (T,N,2)
        "speed": torch.tensor(speed, dtype=torch.float32, device=device),    # (T,N) 원 단위
        "adj": torch.tensor(adj, dtype=torch.float32, device=device),        # (N,N)
        "scaler": scaler, "t_in": t_in, "horizon": horizon, "null_val": null_val,
        "T": T, "N": N, "W": W, "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "starts": {"train": np.arange(0, n_train),
                   "val": np.arange(n_train, n_train + n_val),
                   "test": np.arange(n_train + n_val, W)},
        "edge_density": edge_density,
    }
    return tens


def _gather(tens, starts, device):
    """윈도우 시작 인덱스 배열 → (x, y). x:(B,t_in,N,2) 정규화특징, y:(B,horizon,N) 원단위."""
    import torch
    t_in, horizon = tens["t_in"], tens["horizon"]
    s = torch.as_tensor(starts, dtype=torch.long, device=device)
    ix = s[:, None] + torch.arange(t_in, device=device)[None, :]           # (B,t_in)
    iy = s[:, None] + t_in + torch.arange(horizon, device=device)[None, :]  # (B,horizon)
    x = tens["feat"][ix]      # (B,t_in,N,2)
    y = tens["speed"][iy]     # (B,horizon,N)
    return x, y


def masked_mae_torch(pred, y, null_val):
    """원 단위 masked MAE (null 제외). pred,y: (...,N) 동일 형상."""
    import torch
    mask = (y != null_val).float()
    denom = mask.sum().clamp_min(1.0)
    return (torch.abs(pred - y) * mask).sum() / denom


def train_one(cfg, tens, adj_mode, epochs, batch_size, lr, wd, patience, device, seed):
    import numpy as np
    import torch
    set_seed(seed)  # 공정 비교: 모드마다 동일 시드로 초기화
    # ⚠️ set_seed 는 cudnn.deterministic=True 로 두는데, 이 모델의 Conv1d(큰 배치)에서
    #    결정론적 알고리즘이 ~25배 느려진다(측정 확인). 학습 속도를 위해 benchmark 모드로
    #    되돌린다. 시드(초기화·데이터 순서)는 고정되나 GPU 합성곱은 bitwise 재현 아님(원리 재현 수준).
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    scaler = tens["scaler"]
    mean, std = scaler.mean, scaler.std
    null_val = tens["null_val"]

    net = Model.build_model(
        num_nodes=tens["N"], in_dim=2, out_dim=1, horizon=tens["horizon"],
        hidden=int(cfg["model"]["hidden"]), n_layers=int(cfg["model"]["n_layers"]),
        diffusion_order=int(cfg["model"]["diffusion_order"]),
        dropout=float(cfg["model"]["dropout"]), node_emb_dim=int(cfg["model"]["node_emb_dim"]),
        adj_mode=adj_mode, adj_fixed=tens["adj"],
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    n_params = sum(p.numel() for p in net.parameters())

    train_starts = tens["starts"]["train"]
    val_starts = tens["starts"]["val"]
    rng = np.random.default_rng(seed)
    best_val = float("inf"); best_state = None; best_epoch = -1; bad = 0
    peak_vram = 0.0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        net.train()
        order = rng.permutation(train_starts)
        tot = 0.0; nb = 0
        n_batches = (len(order) + batch_size - 1) // batch_size
        ep_t0 = time.time()
        for i in range(0, len(order), batch_size):
            bs = order[i:i + batch_size]
            x, y = _gather(tens, bs, device)
            pred = net(x)[..., 0] * std + mean          # (B,horizon,N) 원 단위
            loss = masked_mae_torch(pred, y, null_val)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
            if ep == 1 and nb % 100 == 0:
                dt = time.time() - ep_t0
                print(f"    [{adj_mode:8s}] ep1 batch {nb}/{n_batches}  "
                      f"{dt:.1f}s ({1000*dt/nb:.1f} ms/batch)", flush=True)
        # --- validation (masked MAE, 원 단위) ---
        net.eval(); vtot = 0.0; vnb = 0
        with torch.no_grad():
            for i in range(0, len(val_starts), 256):
                bs = val_starts[i:i + 256]
                x, y = _gather(tens, bs, device)
                pred = net(x)[..., 0] * std + mean
                vtot += masked_mae_torch(pred, y, null_val).item(); vnb += 1
        val_mae = vtot / max(vnb, 1)
        if device == "cuda":
            peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 1e9)
        improved = val_mae < best_val - 1e-4
        if improved:
            best_val = val_mae; best_epoch = ep; bad = 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        print(f"    [{adj_mode:8s}] ep {ep:3d}/{epochs}  train_mae={tot/max(nb,1):.4f}  "
              f"val_mae={val_mae:.4f}  best={best_val:.4f}@{best_epoch}  "
              f"vram={peak_vram:.2f}GB  bad={bad}", flush=True)
        if bad >= patience:
            print(f"    [{adj_mode:8s}] early stop @ep{ep} (patience {patience})")
            break
    train_time = time.time() - t0
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, {"n_params": n_params, "best_val_mae": best_val, "best_epoch": best_epoch,
                 "epochs_run": ep, "train_time_sec": round(train_time, 1),
                 "peak_vram_gb": round(peak_vram, 2)}


def evaluate_test(net, tens, horizons, device, batch=256):
    """test split 전체 예측 → 지표. 반환 dict(at_horizons/per_step/mae_divergence)."""
    import numpy as np
    import torch
    scaler = tens["scaler"]; mean, std = scaler.mean, scaler.std
    null_val = tens["null_val"]
    test_starts = tens["starts"]["test"]
    preds = []; gts = []
    net.eval()
    with torch.no_grad():
        for i in range(0, len(test_starts), batch):
            bs = test_starts[i:i + batch]
            x, y = _gather(tens, bs, device)
            pred = net(x)[..., 0] * std + mean            # (B,horizon,N) 원 단위
            preds.append(pred.cpu().numpy()); gts.append(y.cpu().numpy())
    pred = np.concatenate(preds, 0)                        # (n_test,horizon,N)
    gt = np.concatenate(gts, 0)
    p = np.moveaxis(pred, 1, 0)                            # (horizon,n_test,N)
    g = np.moveaxis(gt, 1, 0)
    at_h = M.metrics_at_horizons(p, g, horizons=horizons, null_val=null_val)
    per_step = M.metrics_per_step(p, g, null_val=null_val)
    div = rollout_divergence(per_step["per_step"]["mae"])
    return {"at_horizons": at_h, "per_step": per_step, "mae_divergence": div}


def baseline_scores(tens, horizons):
    """동일 test split 에서 copy-last / seasonal-HA 재계산(기준선과 apples-to-apples)."""
    import numpy as np
    t_in, horizon = tens["t_in"], tens["horizon"]
    null_val = tens["null_val"]
    speed = tens["speed"].cpu().numpy()
    period = 7 * 24 * 12
    test_starts = tens["starts"]["test"]
    target_starts = test_starts + t_in
    gt = np.stack([speed[target_starts + h] for h in range(horizon)], axis=1)     # (n_test,horizon,N)
    last = speed[target_starts - 1]
    pred_copy = np.repeat(last[:, None, :], horizon, axis=1)
    table = B.seasonal_average_table(speed, train_len=tens["n_train"] + t_in, period=period, null_val=null_val)
    pred_ha = B.seasonal_ha_predict(table, target_starts, horizon, period)

    def _score(pred):
        p = np.moveaxis(pred, 1, 0); g = np.moveaxis(gt, 1, 0)
        at_h = M.metrics_at_horizons(p, g, horizons=horizons, null_val=null_val)
        per_step = M.metrics_per_step(p, g, null_val=null_val)
        div = rollout_divergence(per_step["per_step"]["mae"])
        return {"at_horizons": at_h, "per_step": per_step, "mae_divergence": div}
    return {"copy_last": _score(pred_copy), "seasonal_historical_average": _score(pred_ha)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="METR-LA 소형 STGNN 학습 + RQ1 절제")
    p.add_argument("--config", type=Path, default=_TASK_DIR / "config" / "base.yaml")
    p.add_argument("--modes", nargs="+", default=["fixed", "learned", "hybrid", "identity"])
    p.add_argument("--epochs", type=int, default=None, help="config train.epochs 오버라이드")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--tag", default="stgnn", help="run_id 접두")
    args = p.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    import torch
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    horizons = tuple(cfg["metrics"]["horizons"])
    tr = cfg["train"]
    epochs = args.epochs if args.epochs is not None else int(tr["epochs"])
    batch_size = args.batch_size if args.batch_size is not None else int(tr["batch_size"])
    lr = float(tr["lr"]); wd = float(tr["weight_decay"]); patience = int(tr["early_stop_patience"])

    want = args.device or (str(tr.get("device", "cuda")))
    device = "cuda" if (want == "cuda" and torch.cuda.is_available()) else "cpu"
    fell_back = (want == "cuda" and device == "cpu")

    set_seed(seed)
    tens = prepare(cfg, device)
    print(f"[stgnn] device={device}{' (CUDA 미탐지 → CPU 폴백)' if fell_back else ''} "
          f"torch={torch.__version__} seed={seed} epochs={epochs} batch={batch_size}")
    print(f"[stgnn] data (T,N)=({tens['T']},{tens['N']}) windows train/val/test="
          f"{tens['n_train']}/{tens['n_val']}/{tens['n_test']} adj_edge_density={tens['edge_density']:.3f}")

    stgnn_results = {}; stgnn_meta = {}
    for mode in args.modes:
        print(f"[stgnn] === train adj_mode={mode} ===", flush=True)
        net, meta = train_one(cfg, tens, mode, epochs, batch_size, lr, wd, patience, device, seed)
        stgnn_results[mode] = evaluate_test(net, tens, horizons, device)
        stgnn_meta[mode] = meta
        mae = stgnn_results[mode]["at_horizons"]["mae"]
        print(f"[stgnn] {mode}: test MAE @15/30/60 = {mae['h3']:.3f}/{mae['h6']:.3f}/{mae['h12']:.3f}"
              f"  (best_val={meta['best_val_mae']:.3f}@ep{meta['best_epoch']}, "
              f"{meta['train_time_sec']}s, vram {meta['peak_vram_gb']}GB)", flush=True)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()

    baselines = baseline_scores(tens, horizons)

    run_id = datetime.now(timezone.utc).strftime(f"{args.tag}-metr-la-%Y%m%dT%H%M%SZ")
    out_dir = args.out or (_TASK_DIR / "results" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "n/a"

    metrics = {
        "synthetic_dummy": False,
        "task": "urban-traffic-forecasting", "phase": "Phase 2-B — STGNN 학습 + RQ1 절제",
        "note": "SOTA 재현 아님(원리 재현). 결측=0 masked MAE/RMSE/MAPE. 논문값은 reference_only.",
        "dataset": {"name": "METR-LA", "shape_T_N": [tens["T"], tens["N"]],
                    "sample_freq_min": 5, "source": "liyaguang/DCRNN (연구용 공개)"},
        "protocol": {"seq_len_in": tens["t_in"], "horizon": tens["horizon"],
                     "horizons_steps": list(horizons),
                     "horizons_minutes": {f"h{h}": h * 5 for h in horizons},
                     "split_windows": {"train": tens["n_train"], "val": tens["n_val"], "test": tens["n_test"]},
                     "eval_split": "test", "features": ["zscore_speed", "time_of_day"],
                     "loss": "masked_mae(원 단위)", "adj_edge_density": round(tens["edge_density"], 4)},
        "run": {"seed": seed, "git_commit": _git_hash(), "timestamp_utc": run_id.split("-")[-1],
                "epochs_cap": epochs, "batch_size": batch_size, "lr": lr, "weight_decay": wd,
                "early_stop_patience": patience,
                "hardware": {"platform": platform.platform(), "device": device, "gpu": gpu_name,
                             "cuda_fallback_to_cpu": fell_back}},
        "stgnn_test": {m: {**stgnn_results[m], "train_meta": stgnn_meta[m]} for m in args.modes},
        "baselines_test": baselines,
        "reference_only": {
            "note": "DCRNN 논문(Li+ 2018 Table 1) METR-LA test. 우리 소형/축소 설정과 달라 직접 비교 아님.",
            "DCRNN_paper_mae": {"h3": 2.77, "h6": 3.15, "h12": 3.60},
            "GraphWaveNet_paper_mae": {"h3": 2.69, "h6": 3.07, "h12": 3.53},
            "HA_paper_mae": {"h3": 4.16, "h6": 4.16, "h12": 4.16},
        },
    }
    M.save_metrics(metrics, out_dir / "metrics.json")

    # --- summary.md (STGNN + 기준선 통합 표) ---
    def mae_row(name, r, extra=""):
        m = r["at_horizons"]["mae"]; d = r["mae_divergence"]
        return (f"| {name} | {m['h3']:.3f} | {m['h6']:.3f} | {m['h12']:.3f} "
                f"| {r['at_horizons']['rmse']['h12']:.3f} | {r['at_horizons']['mape']['h12']:.2f} "
                f"| {d['slope']:.4f} | {extra} |")
    lines = [
        f"# METR-LA STGNN + 기준선 — {run_id}", "",
        f"- 데이터 METR-LA (T,N)=({tens['T']},{tens['N']}), test {tens['n_test']} 윈도우, 원 단위(mph), masked(null=0)",
        f"- 프로토콜 과거 {tens['t_in']}→미래 {tens['horizon']} 스텝, 특징=[zscore speed, time-of-day]",
        f"- device={device}, seed={seed}, epochs_cap={epochs}, batch={batch_size}, git={_git_hash()[:8]}",
        f"- SOTA 재현 아님(원리 재현). 논문값은 참조용(설정 차이로 직접 비교 아님).", "",
        "| 모델 | MAE@15m | MAE@30m | MAE@60m | RMSE@60m | MAPE@60m(%) | MAE기울기 | 비고 |",
        "|---|---|---|---|---|---|---|---|",
        mae_row("copy-last", baselines["copy_last"], "기준선"),
        mae_row("seasonal-HA", baselines["seasonal_historical_average"], "기준선"),
    ]
    for m in args.modes:
        meta = stgnn_meta[m]
        lines.append(mae_row(f"STGNN ({m})", stgnn_results[m],
                             f"{meta['train_time_sec']}s·{meta['peak_vram_gb']}GB·ep{meta['best_epoch']}"))
    lines += [
        f"| *(참조)* *DCRNN 논문* | *2.77* | *3.15* | *3.60* | *—* | *—* | *—* | *참조용* |", "",
        "> RQ1(고정 vs 학습): 위 STGNN 행들의 fixed vs learned/hybrid MAE 비교.",
        "> RQ2(오차 누적): MAE기울기(스텝당) — 기준선 copy-last 대비 STGNN 이 덜 누적하는지.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[stgnn] 완료 → {out_dir}")
    print("[stgnn] test MAE@15/30/60 요약:")
    print(f"  copy-last   {baselines['copy_last']['at_horizons']['mae']}")
    print(f"  seasonal-HA {baselines['seasonal_historical_average']['at_horizons']['mae']}")
    for m in args.modes:
        print(f"  STGNN {m:8s} {stgnn_results[m]['at_horizons']['mae']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

