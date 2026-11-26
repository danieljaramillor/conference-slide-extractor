"""Generated slide captures with replaced text and additive panels."""

import cv2
import numpy as np


INK = (54, 36, 19)


def text(image, value, origin, size=.53):
    cv2.putText(image, value, origin, cv2.FONT_HERSHEY_SIMPLEX, size, INK, 1, cv2.LINE_AA)


def shared_slide():
    image = np.full((540, 960, 3), 248, np.uint8)
    text(image, 'Clinical evidence and study eligibility', (42, 32), 1.0)
    for row in range(12):
        text(image, 'Evidence collected in the eligible primary cohort', (60, 140 + row * 31))
    return image


def heading_build_stages():
    earlier = shared_slide()
    newer = earlier.copy()
    text(earlier, 'Original eligibility interpretation', (315, 70), .65)
    text(newer, 'Updated evidence classification', (315, 70), .65)
    complete = newer.copy()
    points = np.array([(680, 456), (760, 340), (840, 456)], dtype=np.int32)
    cv2.polylines(complete, [points], True, INK, 3)
    cv2.line(complete, (704, 418), (816, 418), INK, 3)
    return earlier, newer, complete
