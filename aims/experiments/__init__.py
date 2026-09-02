"""Experiment modules for AIMS."""

import os
import wandb


def wandb_init(**kwargs):
    """wandb.init() 래퍼 — 로그인 없이도 실행 가능.

    WANDB_MODE=offline 환경변수 또는 wandb 미로그인 시
    자동으로 offline 모드로 폴백합니다.
    """
    mode = os.environ.get("WANDB_MODE", "online")
    try:
        return wandb.init(mode=mode, **kwargs)
    except Exception:
        print("[W&B] 로그인 정보 없음 → offline 모드로 실행")
        return wandb.init(mode="offline", **kwargs)
