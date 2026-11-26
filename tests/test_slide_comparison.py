"""Review highlights are advisory and preserve capture-specific geometry."""
import unittest

import cv2
import numpy as np

from mixed_fixtures import make_slide
from slide_comparison import compare_captures
from transition_fixtures import slide_build_stages


class SlideComparisonTests(unittest.TestCase):
    def test_new_panel_and_qr_have_localized_highlights_and_complete_side(self):
        base, panel, complete = slide_build_stages()
        for a, b, minimum_top in ((base, panel, .48), (panel, complete, .65)):
            with self.subTest(minimum_top=minimum_top):
                comparison = compare_captures(a, b)
                self.assertEqual(comparison['relationship'], 'right-complete')
                self.assertEqual(comparison['status'], 'ok')
                self.assertEqual(len(comparison['regions']['right']), 1)
                left, top, right, bottom = comparison['regions']['right'][0]
                self.assertGreater(left, .6)
                self.assertGreater(top, minimum_top)
                self.assertLess(right - left, .2)
                self.assertLess(bottom - top, .24)
                self.assertEqual(compare_captures(b, a)['relationship'], 'left-complete')

    def test_different_native_sizes_and_small_crop_shift_support_highlights(self):
        a, _, b = slide_build_stages()
        smaller = cv2.resize(b, (800, 450), interpolation=cv2.INTER_AREA)
        comparison = compare_captures(a, smaller)
        self.assertEqual(comparison['status'], 'ok')
        self.assertEqual(len(comparison['regions']['right']), 2)
        self.assertNotIn(comparison['relationship'], {'left-complete', 'right-complete'})

        matrix = np.float32([[1, 0, 9], [0, 1, 7]])
        shifted = cv2.warpAffine(b, matrix, (960, 540), borderValue=(248, 248, 248))
        comparison = compare_captures(a, shifted)
        left_boxes = comparison['regions']['left']
        right_boxes = comparison['regions']['right']
        self.assertEqual(comparison['status'], 'ok')
        self.assertEqual(len(left_boxes), 2)
        for lbox, rbox in zip(left_boxes, right_boxes):
            self.assertAlmostEqual(rbox[0] - lbox[0], 9 / 960, places=3)
            self.assertAlmostEqual(rbox[1] - lbox[1], 7 / 540, places=3)

    def test_identical_and_different_geometry_do_not_show_false_highlights(self):
        a, _, b = slide_build_stages()
        comparison = compare_captures(a, a.copy())
        self.assertEqual(comparison['relationship'], 'same-content')
        self.assertFalse(comparison['regions']['left'])
        comparison = compare_captures(a, cv2.resize(b, (480, 600)))
        self.assertEqual(comparison['status'], 'uncertain')
        self.assertFalse(comparison['regions']['left'])
        no_alignment = cv2.flip(b, 1)
        comparison = compare_captures(a, no_alignment)
        self.assertEqual(comparison['status'], 'uncertain')
        self.assertFalse(comparison['regions']['right'])

    def test_number_change_stays_separate_and_reports_ocr_conflict(self):
        def slide_metadata(number):
            lines = [
                ('Mixed layout validation', 95), ('Cohort A and cohort B', 410),
                ('Study result: ' + number + ' percent', 600),
            ]
            return {'slide_width': 1920, 'slide_height': 1080, 'ocr': {
                'status': 'ok', 'median_text_height': 25, 'lines': [
                    {'text': value, 'score': .99, 'height': 25,
                     'box': [[65, y], [590, y], [590, y + 25], [65, y + 25]]}
                    for value, y in lines
                ],
            }}

        left, right = make_slide(number='47.8'), make_slide(number='42.8')
        visual = compare_captures(left, right)
        self.assertTrue(visual['regions']['right'])
        self.assertNotIn(visual['relationship'], {'left-complete', 'right-complete', 'same-content'})
        verified = compare_captures(left, right, slide_metadata('47.8'), slide_metadata('42.8'))
        self.assertTrue(verified['numeric_conflict'])
        self.assertEqual(verified['relationship'], 'content-replaced')

    def test_fine_native_content_vetoes_completion_even_with_new_panel(self):
        base, panel, _ = slide_build_stages()
        base, panel = (cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_NEAREST)
                       for image in (base, panel))
        # The earlier capture has a small marker far from the newly added panel.
        base[100:104, 1500:1504] = (15, 45, 190)
        comparison = compare_captures(base, panel)
        self.assertTrue(comparison['regions']['right'])
        self.assertNotIn(comparison['relationship'], {'left-complete', 'right-complete'})

    def test_replaced_chart_has_evidence_on_both_captures(self):
        left, _, _ = slide_build_stages()
        right = left.copy()
        cv2.line(left, (65, 414), (365, 367), (19, 200, 250), 10)
        cv2.line(right, (65, 377), (365, 422), (190, 35, 140), 10)
        comparison = compare_captures(left, right)
        self.assertEqual(comparison['status'], 'ok')
        self.assertEqual(comparison['relationship'], 'content-replaced')
        self.assertTrue(comparison['regions']['left'])
        self.assertTrue(comparison['regions']['right'])


if __name__ == '__main__':
    unittest.main()
