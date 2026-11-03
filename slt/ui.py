"""Small OpenCV helpers shared by the interactive scripts."""
from __future__ import annotations

import platform

import cv2
import numpy as np


def open_camera(index: int = 0, width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index}. Try --camera 1, close other apps using the webcam, "
                           "or check the OS camera permission for your terminal / IDE.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def put_text(img, text, org, scale=0.7, color=(255, 255, 255), thick=2, bg=(0, 0, 0)):
    """Text with a dark outline so it is readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, bg, thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def panel(img, height: int = 90, alpha: float = 0.55, bottom: bool = False):
    """Darken a band at the top (or bottom) of the frame for overlay text."""
    h, w = img.shape[:2]
    y0, y1 = (h - height, h) if bottom else (0, height)
    band = img[y0:y1].copy()
    img[y0:y1] = cv2.addWeighted(band, 1 - alpha, np.zeros_like(band), alpha, 0)


def wrap_text(text: str, max_chars: int = 60) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines or [""]
