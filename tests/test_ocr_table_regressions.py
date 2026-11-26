"""Protect small printed changes while ignoring common table OCR artifacts."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from mixed_fixtures import tiny_numeric_captures
from slide_ocr import local_numeric_evidence, read_text, text_evidence
from test_ocr_numeric import _capture


class TableRegressions(unittest.TestCase):
    def test_comma_spacing_cannot_split_a_count_and_percentage(self):
        compact = _capture(('17/24,70.8%', 160, 20))
        spaced = _capture(('17/24, 70.8%', 160, 20))
        changed = _capture(('17/24, 70.3%', 160, 20))
        self.assertFalse(text_evidence(compact, spaced)['numeric_conflict'])
        self.assertTrue(text_evidence(compact, changed)['numeric_conflict'])

    def test_a_sign_change_is_a_numeric_change(self):
        positive = _capture(('12.5%', 160, 20))
        negative = _capture(('−12.5%', 160, 20))
        self.assertTrue(text_evidence(positive, negative)['numeric_conflict'])

    def test_dropped_legend_label_does_not_borrow_the_row_above(self):
        first = _capture(('16.5 days', 160, 18))
        second = _capture(('31.5 days', 172, 18))
        self.assertFalse(text_evidence(first, second)['numeric_conflict'])

    def test_cursor_occlusion_and_actual_leading_digit_removal_are_distinct(self):
        shorter = _capture(('0.8%', 160, 20))
        longer = _capture(('70.8%', 160, 20))
        removed = np.full((800, 1440, 3), 252, np.uint8)
        ink = (25, 25, 25)
        cv2.putText(removed, '0.8%', (658, 177), cv2.FONT_HERSHEY_SIMPLEX, .65, ink, 1, cv2.LINE_AA)
        clear = removed.copy()
        cv2.putText(clear, '7', (645, 177), cv2.FONT_HERSHEY_SIMPLEX, .65, ink, 1, cv2.LINE_AA)
        covered = clear.copy()
        cursor = np.array([(647, 163), (647, 175), (650, 172), (653, 178),
                           (655, 176), (652, 172), (657, 172)], np.int32)
        cv2.fillPoly(covered, [cursor], (253, 253, 253), cv2.LINE_AA)
        cv2.polylines(covered, [cursor], True, (44, 44, 44), 1, cv2.LINE_AA)
        artifact = text_evidence(shorter, longer, native_pair=lambda: (covered, clear))
        self.assertFalse(artifact['numeric_conflict'])
        self.assertEqual(artifact['numeric_occlusions'], ['a'])
        self.assertTrue(text_evidence(shorter, longer, native_pair=lambda: (removed, clear))['numeric_conflict'])

    def test_local_reread_finds_superscript_missed_by_full_slide_ocr(self):
        first, second = (image for _, image in tiny_numeric_captures())
        self.assertEqual(local_numeric_evidence(first, first), [])
        self.assertEqual(local_numeric_evidence(first, second), [['2', '3']])
        # Run after the local read to verify the shared engine returns line boxes again.
        with TemporaryDirectory() as directory:
            for image, glyph in ((first, '2'), (second, '3')):
                ocr = read_text(image, Path(directory))
                self.assertTrue(ocr['lines'])
                self.assertNotIn(glyph, [line['text'].strip() for line in ocr['lines']])


if __name__ == '__main__':
    unittest.main()
