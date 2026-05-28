from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: str | Path) -> None:
    ensure_parent(path)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def file_size_mb(path: str | Path) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(np.array(values), p))


def init_wandb_run(
    use_wandb: bool,
    *,
    project: str,
    run_name: str | None = None,
    entity: str | None = None,
    mode: str = "online",
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
):
    if not use_wandb:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "Không tìm thấy thư viện wandb. Hãy cài dependencies trong môi trường HocSau trước khi bật --use_wandb."
        ) from exc

    return wandb.init(
        project=project,
        name=run_name,
        entity=entity,
        mode=mode,
        config=config,
        tags=tags,
    )


def wandb_run_info(run) -> dict[str, Any]:
    if run is None:
        return {"enabled": False}

    project_attr = getattr(run, "project", None)
    if isinstance(project_attr, str):
        project = project_attr
    else:
        project = getattr(project_attr, "name", None)

    return {
        "enabled": True,
        "run_id": getattr(run, "id", None),
        "run_name": getattr(run, "name", None),
        "run_url": getattr(run, "url", None),
        "project": project,
    }
