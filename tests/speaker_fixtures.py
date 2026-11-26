"""Reusable offline camera frame built around the licensed test portrait."""
from pathlib import Path

import cv2
import numpy as np


def load_portrait():
    path = Path(__file__).with_name('fixtures') / 'speaker-portrait.jpg'
    return cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)


def camera_capture():
    portrait = load_portrait()
    height, width = portrait.shape[:2]
    stage = np.full((360, 640, 3), 18, np.uint8)
    stage[80:80+height, 272:272+width] = portrait
    cv2.putText(stage, 'Conference moderator', (20, 328), cv2.FONT_HERSHEY_SIMPLEX, .65,
                (245, 245, 245), 2, cv2.LINE_AA)
    screenshot = np.full((600, 1000, 3), 70, np.uint8)
    screenshot[95:520, 60:820] = cv2.resize(stage, (760, 425), interpolation=cv2.INTER_CUBIC)
    return screenshot
