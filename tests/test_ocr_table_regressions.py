"""Protect small printed changes while ignoring common table OCR artifacts."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from mixed_fixtures import tiny_numeric_captures
from slide_ocr import local_numeric_evidence, read_text, text_evidence
from test_ocr_numeric import _capture


class TableRegressions(unittest.TestCase):
    def test_bibliography_dash_noise_does_not_change_cited_pages(self):
        first = _capture(('J Oncol 2024: 10:25–-31', 160, 20))
        second = _capture(('J Oncol 2024: 10:25–31', 160, 20))
        changed = _capture(('J Oncol 2024: 10:25–32', 160, 20))
        self.assertFalse(text_evidence(first, second)['numeric_conflict'])
        self.assertTrue(text_evidence(first, changed)['numeric_conflict'])

    def test_citation_marker_requires_matching_native_ink(self):
        first = _capture(('Study evidence{1', 160, 20))
        second = _capture(('Study evidence^11', 160, 20))
        clear = np.full((800, 1440, 3), 248, np.uint8)
        artifact = clear.copy()
        artifact[177, 690] = 218
        changed = clear.copy()
        changed[169:174, 687:690] = 30
        self.assertFalse(text_evidence(first, second, native_pair=lambda: (artifact, clear))['numeric_conflict'])
        self.assertTrue(text_evidence(first, second, native_pair=lambda: (changed, clear))['numeric_conflict'])

    def test_fused_citation_and_numbered_affiliation_preserve_printed_changes(self):
        alternatives = (
            ('2 years on 1L ET + CDK4/69', '2 years on 1L ET + CDK4/6i9'),
            ('1Central Lab, District, City, Country Ltd.', '14Central Lab, District, City, Country Ltd.'),
        )
        clear = np.full((800, 1440, 3), 248, np.uint8)
        artifact, changed = clear.copy(), clear.copy()
        artifact[177, 690] = 209
        changed[169:174, 687:690] = 30
        for original, other in alternatives:
            with self.subTest(original=original):
                first, second = (_capture((value, 160, 20)) for value in (original, other))
                self.assertFalse(text_evidence(first, second,
                                               native_pair=lambda: (artifact, clear))['numeric_conflict'])
                self.assertTrue(text_evidence(first, second,
                                              native_pair=lambda: (changed, clear))['numeric_conflict'])

    def test_unreadable_native_reread_preserves_disputed_print_for_review(self):
        first, second = (_capture((value, 160, 20)) for value in ('4.0', '40'))
        clear = np.full((800, 1440, 3), 248, np.uint8)
        changed = clear.copy()
        changed[165:167, 667:669] -= 12
        with patch('slide_ocr._recognize_cell', side_effect=[RuntimeError('unreadable'), ('40', .97)]):
            evidence = text_evidence(first, second, native_pair=lambda: (changed, clear), reread_numeric=True)
        self.assertTrue(evidence['numeric_conflict'])
        self.assertEqual(len(evidence['numeric_reviews']), 1)
        review = evidence['numeric_reviews'][0]
        self.assertEqual((review['read_a'], review['read_b']), ('', '40'))
        self.assertEqual(review['bounds_a'], [637, 157, 703, 183])

    def test_agreeing_rereads_cannot_erase_a_faint_native_decimal(self):
        first, second = (_capture((value, 160, 20)) for value in ('4.0', '40'))
        clear = np.full((800, 1440, 3), 248, np.uint8)
        changed = clear.copy()
        changed[165:167, 667:669] -= 12
        with patch('slide_ocr._recognize_cell', return_value=('40', .99)):
            evidence = text_evidence(first, second, native_pair=lambda: (changed, clear), reread_numeric=True)
        self.assertTrue(evidence['numeric_conflict'])
        self.assertEqual(len(evidence['numeric_reviews']), 1)

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

    def test_local_decimal_omission_requires_unchanged_print_below_the_cursor(self):
        clear = np.full((800, 1440, 3), 252, np.uint8)
        cv2.putText(clear, '0.70', (420, 353), cv2.FONT_HERSHEY_SIMPLEX, .55,
                    (20, 20, 20), 1, cv2.LINE_AA)
        covered = clear.copy()
        cursor = np.array([(428, 326), (428, 338), (431, 335), (434, 341),
                           (436, 339), (433, 335), (437, 335)], np.int32)
        cv2.fillPoly(covered, [cursor], (253, 253, 253), cv2.LINE_AA)
        cv2.polylines(covered, [cursor], True, (44, 44, 44), 1, cv2.LINE_AA)
        removed = covered.copy()
        removed[350:355, 431:437] = 252

        for capture, expected in ((covered, ['a']), (removed, [])):
            with self.subTest(decimal_missing=not expected):
                unresolved, occlusions = [], []
                with patch('slide_ocr._recognize_cell', return_value=('0.70', .99)) as reader:
                    reader.side_effect = [('070', .86), ('0.70', .99)] * 4
                    local_numeric_evidence(capture, clear, occlusions=occlusions, unresolved=unresolved)
                self.assertEqual(occlusions, expected)
                self.assertEqual(bool(unresolved), not expected)

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
