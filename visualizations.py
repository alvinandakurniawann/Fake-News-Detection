# visualizations.py
"""
Sorotan kata berpengaruh (kelas CSS di ui.py).
Meter keyakinan & bar probabilitas kini komponen HTML di ui.py
(lebih ringan, tanpa toolbar) — fungsi Plotly lama sudah dihapus.
"""

import re
from typing import Dict, List


def _tier(weight: float) -> str:
    """Tiga tingkat intensitas berdasarkan |bobot|."""
    a = abs(weight)
    if a >= 0.6:
        return "3"
    if a >= 0.25:
        return "2"
    return "1"


def highlight_important_words(
    text: str,
    important_words: List[Dict],
    preprocessor=None,
    preprocessing_steps=None,
) -> str:
    """Sorot kata berpengaruh dengan kelas CSS + tooltip bobot.

    Oranye = mengarah ke hoaks, cyan = mengarah ke valid.
    Tanda tangan fungsi dipertahankan.
    """
    if not important_words or not text:
        return text

    if 'class="hl ' in text or "class='hl " in text:
        return text

    word_highlights = {}
    for word_info in important_words:
        word = word_info.get("word", "").strip()
        if not word or len(word) < 2:
            continue
        weight = word_info.get("weight", 0)
        side = "f" if weight > 0 else "r"
        word_highlights[word.lower()] = {
            "cls": f"hl hl-{side}{_tier(weight)}",
            "tip": f"Weight {weight:+.3f} — {'points toward fake' if weight > 0 else 'points toward real'}",
        }

    if not word_highlights:
        return text

    words_sorted = sorted(word_highlights.keys(), key=len, reverse=True)
    pattern = re.compile(
        r"(?<![\w-])(" + "|".join(map(re.escape, words_sorted)) + r")(?![\w-])",
        re.IGNORECASE,
    )

    result_parts = []
    last_end = 0
    for match in pattern.finditer(text):
        result_parts.append(text[last_end : match.start()])
        matched_word = match.group(1)
        info = word_highlights.get(matched_word.lower())
        if info:
            result_parts.append(
                f'<span class="{info["cls"]}" title="{info["tip"]}">{matched_word}</span>'
            )
        else:
            result_parts.append(matched_word)
        last_end = match.end()
    result_parts.append(text[last_end:])
    return "".join(result_parts)
