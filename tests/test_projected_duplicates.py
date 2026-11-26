"""Verify real slide content survives projected-capture deduplication."""

import unittest

import cv2
import numpy as np

from slide_comparison import tint_duplicate


def _table(value='47.8', recolor=False):
    image = np.full((756, 1344, 3), 246, np.uint8)
    font, ink = cv2.FONT_HERSHEY_SIMPLEX, (44, 50, 61)
    cv2.putText(image, 'Population and clinical response', (54, 90),
                cv2.FONT_HERSHEY_DUPLEX, 1.62, (95, 53, 28), 3, cv2.LINE_AA)
    cv2.line(image, (54, 130), (1290, 130), (166, 100, 38), 3)
    labels = ('Control arm', 'Pooling total', 'Treatment', 'Expected', 'Follow-up', 'Independent',
              'Evaluable', 'Median', 'Included', 'Eligibility', 'Analysis', 'Secondary',
              'Cohort R', 'Cohort D', 'Cohort A', 'Cohort J', 'Phase 2', 'Phase 3')
    for i, label in enumerate(labels):
        y, x = 200 + 62 * (i % 9), 65 + 650 * (i // 9)
        cv2.rectangle(image, (x - 18, y - 38), (x + 509, y + 12), (225 - i * 2, 239 - i, 244), -1)
        cv2.putText(image, label, (x + 12, y), font, .76, (49, 60, 72), 2, cv2.LINE_AA)
        cell = value if i == 9 else f'{(i + 7) * 18 % 91 + .1 * i:.1f}'
        cv2.putText(image, cell, (x + 270, y), font, .67, ink, 2, cv2.LINE_AA)
        color = (40, 98, 224) if i % 3 == 0 else (179, 153, 70)
        if recolor and i == 8:
            color = (44, 133, 157)  # Keep the changed bar's luminance almost constant.
        cv2.rectangle(image, (x + 404, y - 22), (x + 501, y - 9), color, -1)
        cv2.line(image, (x + 200, y - 25), (x + 200, y + 6), (119, 119, 119), 1)
    cv2.putText(image, 'Source: summary table 7', (50, 718),
                cv2.FONT_HERSHEY_PLAIN, 1.3, (50, 50, 50), 2, cv2.LINE_AA)
    return image


class ProjectedDuplicateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = _table()
        size = (720, 405)
        smaller = cv2.resize(cls.clean, size, interpolation=cv2.INTER_AREA)
        transform = np.float32([[.997, -.004, 4.5], [-.001, .995, 1.7], [0, 0, 1]])
        warped = cv2.warpPerspective(smaller, transform, size, borderMode=cv2.BORDER_REPLICATE)
        cls.projected = np.clip(warped.astype(np.float32) * [.88, .92, .76] + [28, 13, 3],
                                0, 255).astype(np.uint8)

    def test_global_projection_tint_keeps_one_capture(self):
        self.assertTrue(tint_duplicate(self.clean, self.projected))

    def test_changed_single_glyph_or_decimal_remains_separate(self):
        for value in ('47.6', '478'):
            with self.subTest(value=value):
                self.assertFalse(tint_duplicate(_table(value), self.projected))

    def test_changed_chart_color_with_unchanged_shape_is_distinct(self):
        self.assertFalse(tint_duplicate(_table(recolor=True), self.projected))
