import unittest

import cv2
import numpy as np

from presenter_fixtures import SLIDE_BOUNDS, presenter_capture_sequence
from slide_regions import confirmed_presenter_crops, extract, presenter_layout


class PresenterCropTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = [image for _, image in presenter_capture_sequence()]
        cls.layouts = [presenter_layout(image) for image in cls.frames]

    def test_recovers_native_bounds_throughout_a_moving_presenter_sequence(self):
        crops = confirmed_presenter_crops(self.layouts)
        self.assertEqual(sorted(crops), list(range(8)))
        x1, y1, x2, y2 = SLIDE_BOUNDS
        for index, crop in crops.items():
            with self.subTest(index=index):
                image = self.frames[index]
                np.testing.assert_array_equal(extract(image, {'crop': crop}), image[y1:y2, x1:x2])

    def test_supports_the_presenter_on_the_other_side(self):
        mirrored = [cv2.flip(frame, 1) for frame in self.frames[:3]]
        crops = confirmed_presenter_crops([presenter_layout(image) for image in mirrored])
        self.assertEqual(sorted(crops), list(range(3)))
        x1, y1, x2, y2 = SLIDE_BOUNDS
        width = mirrored[0].shape[1]
        for index, crop in crops.items():
            np.testing.assert_array_equal(extract(mirrored[index], {'crop': crop}),
                                          mirrored[index][y1:y2, width - x2:width - x1])

    def test_crop_requires_motion_and_stops_at_a_gap(self):
        still = self.layouts[0]
        self.assertIsNotNone(still)
        self.assertEqual(confirmed_presenter_crops([still, still, still]), {})
        self.assertEqual(confirmed_presenter_crops(self.layouts[:2]), {})
        layouts = [*self.layouts[:3], None, still, still, still]
        self.assertEqual(sorted(confirmed_presenter_crops(layouts)), [0, 1, 2])

    def test_adjacent_trusted_pane_crops_recover_an_isolated_miss_without_motion(self):
        still = self.layouts[0]
        layouts = [still, still, still]
        known = [still.crop, None, still.crop]
        self.assertEqual(confirmed_presenter_crops(layouts, known), {1: still.crop})
        self.assertEqual(confirmed_presenter_crops(layouts, [still.crop, None, None]), {})
        self.assertEqual(confirmed_presenter_crops([None, still, still], known), {})

        different_crop = (.07, .1, .80, .88)
        self.assertEqual(confirmed_presenter_crops(layouts, [still.crop, None, different_crop]), {})
        self.assertEqual(confirmed_presenter_crops([still, self.layouts[7], still], known), {})

    def test_crop_evidence_does_not_carry_over_to_different_broadcast_chrome(self):
        different_show = self.frames[0].copy()
        different_show[:150] = (115, 80, 16)
        changed = presenter_layout(different_show)
        self.assertIsNotNone(changed)
        layouts = [*self.layouts[:3], changed, changed, changed]
        self.assertEqual(sorted(confirmed_presenter_crops(layouts)), [0, 1, 2])

    def test_preserves_in_slide_portraits_and_independent_graphics(self):
        self.assertIsNone(self.layouts[-1])
        self.assertIsNone(presenter_layout(np.zeros((18, 32, 3), np.uint8)))
        independent_panel = self.frames[0].copy()
        independent_panel[155:480, 1461:1900] = 242
        cv2.putText(independent_panel, 'Secondary endpoint', (1475, 278),
                    cv2.FONT_HERSHEY_SIMPLEX, .9, (20, 30, 45), 2)
        self.assertIsNone(presenter_layout(independent_panel))

        dark_slide = self.frames[0].copy()
        dark_slide[:155] = 17
        dark_slide[959:] = 17
        cv2.putText(dark_slide, 'Study methods and personnel', (40, 93),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.7, (238, 238, 238), 2)
        self.assertIsNone(presenter_layout(dark_slide))


if __name__ == '__main__':
    unittest.main()
