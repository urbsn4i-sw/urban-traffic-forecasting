"""전 과제 공통 시드 고정 유틸 (PROJECT_GUIDELINE.md §3).

모든 스크립트 시작부에서 `set_seed(cfg.seed)`를 호출한다.
재현성 우선: random / numpy / torch(+cudnn.deterministic) 일괄 설정.
"""
from __future__ import annotations

import os
import random


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """random·numpy·torch 시드를 일괄 고정. torch/numpy 미설치여도 안전.

    Returns 적용한 seed (로그 기록용).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np  # noqa: PLC0415
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch  # noqa: PLC0415
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
