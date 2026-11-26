"""Check row association and confirm OCR disputes against unscaled pixels."""

import unittest

import numpy as np

from slide_ocr import text_evidence


def _capture(*cells):
    def line(text, x, y, width, height):
        return {'text': text, 'score': .99, 'height': height,
                'box': [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]}

    lines = [line('Treatment outcomes by group', 80, 48, 420, 24),
             *(line(value, 640, top, 60, height) for value, top, height in cells)]
    return {'slide_height': 800, 'slide_width': 1440,
            'ocr': {'lines': lines, 'text': '\n'.join(entry['text'] for entry in lines)}}


class NumericEvidenceTests(unittest.TestCase):
    def test_identical_rows_match_even_when_their_centers_are_close(self):
        a = _capture(('4 (80.0)', 160, 20), ('1 (20.0)', 177, 20))
        b = _capture(('1 (20.0)', 178, 20), ('4 (80.0)', 161, 20))
        self.assertFalse(text_evidence(a, b)['numeric_conflict'])

    def test_real_change_matches_by_position_even_if_value_equals_next_row(self):
        a = _capture(('4 (80.0)', 160, 20), ('1 (20.0)', 177, 20))
        b = _capture(('1 (20.0)', 161, 20), ('1 (20.0)', 178, 20))
        self.assertEqual(text_evidence(a, b)['numeric_changes'], [['4 (80.0)', '1 (20.0)']])
        self.assertEqual(text_evidence(b, a)['numeric_changes'], [['1 (20.0)', '4 (80.0)']])

    def test_joined_ocr_box_does_not_split_unchanged_native_cell(self):
        a = _capture(('2', 561, 24))
        b = _capture(('23', 559, 40))
        left = np.full((800, 1440, 3), 244, np.uint8)
        left[565:579, 654:660] = 60
        left[582:595, 654:660] = 60
        right = left.copy()
        right[575, 669] = 228  # Isolated JPEG-like difference inside the text box.
        right[609:621, 654:660] = 60  # The pointer moved outside the contested cell.
        self.assertTrue(text_evidence(a, b)['numeric_conflict'])
        self.assertFalse(text_evidence(a, b, native_pair=lambda: (left, right))['numeric_conflict'])
        self.assertTrue(text_evidence(a, b, native_pair=lambda: (left, right[:-1]))['numeric_conflict'])

    def test_native_check_keeps_a_tiny_decimal_or_faint_glyph_change(self):
        a = _capture(('4.0', 185, 20))
        b = _capture(('40', 185, 20))
        left = np.full((800, 1440, 3), 244, np.uint8)
        for delta, glyph in [(80, (1, 1)), (12, (2, 2))]:
            with self.subTest(delta=delta, glyph=glyph):
                right = left.copy()
                right[197:197 + glyph[0], 674:674 + glyph[1]] -= delta
                self.assertTrue(text_evidence(a, b, native_pair=lambda: (left, right))['numeric_conflict'])


if __name__ == '__main__':
    unittest.main()
