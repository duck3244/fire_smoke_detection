"""InferenceEngine 싱글톤 — 앱 수명 동안 모델 1회 로드."""
from __future__ import annotations

import threading
from pathlib import Path

from inference_engine import InferenceEngine

_lock = threading.Lock()
_engine: InferenceEngine | None = None


def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                eng = InferenceEngine(confidence=0.3, iou_threshold=0.45)
                if not eng.load_model():
                    raise RuntimeError("InferenceEngine load_model() 실패 — best.pt 확인")
                _engine = eng
    return _engine


def reset_engine() -> None:
    """테스트/재로드용."""
    global _engine
    with _lock:
        _engine = None
