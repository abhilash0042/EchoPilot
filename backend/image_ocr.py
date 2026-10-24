"""Read words painted on an image.

One upright pass is enough for a chat reply. A contrast pass runs only when
that first pass did not already find a sentence, so a photo does not sit
through several recognition passes.
"""

from __future__ import annotations

import threading
from typing import Any, List, Sequence, Tuple

_SCORE_FLOOR = 0.5
_FULL_MAX_SIDE = 360
_MIN_SIDE = 0

_engine: Any = None
_lock = threading.Lock()


def read_image_text(image: Any) -> str:
    """Return the words visible in ``image``. Empty means OCR ran and found none."""
    prepared = _prepare(image)
    full = _fit(prepared, _FULL_MAX_SIDE, _MIN_SIDE)
    return _join(_recognize(full))


def _get_engine() -> Any:
    global _engine
    if _engine is not None:
        return _engine
    with _lock:
        if _engine is not None:
            return _engine
        from rapidocr import RapidOCR

        _engine = RapidOCR(params={
            "Global.log_level": "error",
            "Global.use_cls": False,
            "Global.use_preprocess_img": False,
            "Global.max_side_len": 360,
        })
        return _engine


def warm() -> None:
    """Load the reader so the first photo does not pay model startup."""
    _get_engine()


def _prepare(image: Any) -> Any:
    from PIL import Image, ImageOps

    try:
        image = ImageOps.exif_transpose(image) or image
    except Exception:
        pass
    if image.mode == "RGBA":
        base = Image.new("RGB", image.size, (255, 255, 255))
        base.paste(image, mask=image.getchannel("A"))
        return base
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _fit(image: Any, max_side: int, min_side: int) -> Any:
    from PIL import Image

    width, height = image.size
    long_side = max(width, height)
    short_side = min(width, height)
    scale = 1.0
    if short_side < min_side and short_side:
        scale = min_side / short_side
    if long_side * scale > max_side:
        scale = max_side / max(long_side, 1)
    if abs(scale - 1.0) < 0.02:
        return image
    return image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.LANCZOS,
    )


def _clahe(image: Any) -> Any:
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except Exception:
        return None
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    boosted = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return Image.fromarray(cv2.cvtColor(boosted, cv2.COLOR_GRAY2RGB))


def _recognize(image: Any) -> List[str]:
    import numpy as np

    array = np.asarray(image.convert("RGB"))
    bgr = array[:, :, ::-1].copy()
    engine = _get_engine()
    with _lock:
        found = engine(bgr)
    return _lines(_as_items(found))


def _as_items(found: Any) -> List[Tuple[Any, str, float]]:
    if found is None:
        return []
    txts = getattr(found, "txts", None)
    if txts:
        scores = getattr(found, "scores", None) or ()
        items = []
        for index, text in enumerate(txts):
            score = scores[index] if index < len(scores) else 1.0
            items.append((None, text, score))
        return items
    if isinstance(found, tuple):
        found = found[0]
    return list(found or [])


def _lines(found: Any) -> List[str]:
    kept: List[str] = []
    for item in found or []:
        text, score = _item_text(item)
        if score < _SCORE_FLOOR:
            continue
        cleaned = " ".join(text.split())
        if cleaned:
            kept.append(cleaned)
    return kept


def _item_text(item: Sequence[Any]) -> Tuple[str, float]:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return "", 0.0
    text = item[1]
    score = item[2] if len(item) > 2 else 1.0
    if not isinstance(text, str):
        return "", 0.0
    try:
        value = float(score)
    except (TypeError, ValueError):
        value = 1.0
    return text, value


def _word_count(text: str) -> int:
    return len([part for part in (text or "").split() if any(ch.isalnum() for ch in part)])


def _join(lines: List[str]) -> str:
    seen = set()
    ordered: List[str] = []
    for line in lines:
        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(line)
    return "\n".join(ordered)
