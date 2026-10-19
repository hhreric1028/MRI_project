"""Shared paths and provenance helpers for isolated DWI_V2 experiments."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


V2_ROOT = Path(__file__).resolve().parents[1]
V1_ROOT = V2_ROOT.parent / "DWI_V1"
PYTHON = Path(r"C:\Users\User\anaconda3\envs\MRI_DWI\python.exe")

V1_RAW = V1_ROOT / "datasets/ISLES2022/nnUNet_raw/Dataset001_ISLES2022_DWI"
V1_PP = V1_ROOT / "datasets/ISLES2022/processed/Dataset001_ISLES2022_DWI"
V1_P3_RAW = V1_ROOT / "datasets/ISLES2022/nnUNet_raw/Dataset013_ISLES2022_DWI_ClipZScore"

V2_DATASETS = V2_ROOT / "datasets"
V2_RAW_ROOT = V2_DATASETS / "nnUNet_raw"
V2_PP_ROOT = V2_DATASETS / "nnUNet_preprocessed"
V2_RESULTS_ROOT = V2_DATASETS / "nnUNet_results"
E1_DATASET_ID = 101
E1_DATASET_NAME = "Dataset101_ISLES2022_DWI_Original_P3a"
E1_RAW = V2_RAW_ROOT / E1_DATASET_NAME
E1_PP = V2_PP_ROOT / E1_DATASET_NAME

B0_100_PRED = (
    V1_ROOT
    / "experiments/recall_screen100_seed2022_r2/B0_R2022_screen100/train/results"
    / "Dataset001_ISLES2022_DWI/nnUNetTrainer_B0R2022_Screen100__nnUNetPlans__3d_fullres/fold_0/validation"
)
B0_100_METRICS = V1_ROOT / "experiments/recall_screen100_seed2022_r2/B0_R2022_screen100_seed2022/metrics"
if not B0_100_METRICS.exists():
    B0_100_METRICS = V1_ROOT / "experiments/recall_screen100_seed2022_r2/B0_R2022_screen100/metrics"

E1_EXPERIMENT = V2_ROOT / "experiments/V2_E1_original_p3a_dual"
S1_EXPERIMENT = V2_ROOT / "experiments/V2_S1_low_salience_sampling"
S1_HARD = S1_EXPERIMENT / "hard_regions"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def digest(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def ensure_v2(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(V2_ROOT.resolve()):
        raise RuntimeError(f"Refusing to write outside DWI_V2: {resolved}")
    return resolved


def save_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path = ensure_v2(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False, ensure_ascii=False)
        handle.write("\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=V1_ROOT, text=True, encoding="utf-8"
    ).strip()


def source_hashes(paths: list[Path]) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {missing}")
    return {str(path.resolve()): digest(path) for path in paths}


def assert_hashes(expected: dict[str, str]) -> None:
    changed = [path for path, expected_hash in expected.items() if not Path(path).is_file() or digest(Path(path)) != expected_hash]
    if changed:
        raise RuntimeError(f"Frozen source changed: {changed}")


def v1_guard_manifest() -> dict[str, str]:
    paths = [
        V1_RAW / "dataset.json",
        V1_PP / "nnUNetPlans.json",
        V1_PP / "dataset_fingerprint.json",
        V1_PP / "splits_final.json",
        V1_ROOT / "scripts/prepare_p3a_dataset.py",
        V1_ROOT / "scripts/recall_trainers.py",
        V1_ROOT / "scripts/screen100_trainers.py",
        V1_ROOT / "scripts/evaluate_experiment.py",
    ]
    return source_hashes(paths)

