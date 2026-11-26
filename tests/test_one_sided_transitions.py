"""One-sided transition detection preserves content without a verified clean source."""

import unittest

import cv2
import numpy as np

from slide_dedupe import features
from slide_transitions import mark_crossfades, mark_one_sided_transitions
from transition_fixtures import one_sided_sequence


def _items(frames):
    return [
        {
            'id': index, 'name': f'talk/frame_{index:02}.png',
            'crop_mode': 'full', 'crop': (0, 0, 1, 1),
            'gray': features(frame)[1],
        }
        for index, frame in enumerate(frames)
    ]


class OneSidedTransitionTests(unittest.TestCase):
    def setUp(self):
        fixture, self.outgoing = one_sided_sequence()
        self.frames = [frame for _, frame in fixture]

    def _check(self, frames, read_capture=None):
        items = _items(frames)
        mark_crossfades(items)
        mark_one_sided_transitions(items, read_capture=read_capture)
        return items

    def test_unique_outgoing_content_remains_available(self):
        items = self._check(self.frames, lambda item: self.frames[item['id']])
        review = items[2]['transition_review']
        self.assertEqual(review['kind'], 'possible-one-sided-blend')
        self.assertEqual((review['after'], review['support']), (3, 4))
        self.assertNotIn('transition', items[2])
        self.assertNotIn('crossfade', items[2])

    def test_confirmed_remote_clean_source_omits_blend(self):
        frames = [*self.frames, self.outgoing, self.outgoing]
        items = self._check(frames, lambda item: frames[item['id']])
        evidence = items[2]['transition']
        self.assertEqual(evidence['kind'], 'one-sided-crossfade')
        self.assertEqual((evidence['before'], evidence['after']), (5, 3))
        self.assertNotIn('transition_review', items[2])

    def test_absent_or_invalid_native_proof_keeps_review_warning(self):
        frames = [*self.frames, self.outgoing, self.outgoing]

        def read_error(item):
            raise OSError(item['name'])

        readers = (
            None,
            read_error,
            lambda item: None if item['id'] == 2 else frames[item['id']],
            lambda item: frames[item['id']][1:] if item['id'] == 2 else frames[item['id']],
        )
        for reader in readers:
            with self.subTest(reader=reader):
                items = self._check(frames, reader)
                self.assertIn('transition_review', items[2])
                self.assertNotIn('transition', items[2])

    def test_equal_grayscale_can_hide_unique_native_color(self):
        modified = self.frames[2].copy()
        # A small blue-channel change against white is indistinguishable in
        # grayscale here, but is independent information in the native image.
        modified[510:520, 200:225] = (190, 255, 255)
        cv2_gray = lambda frame: cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(cv2_gray(modified), cv2_gray(self.frames[2]))
        frames = [*self.frames[:2], modified, *self.frames[3:], self.outgoing, self.outgoing]
        items = self._check(frames, lambda item: frames[item['id']])
        self.assertIn('transition_review', items[2])
        self.assertNotIn('transition', items[2])

    def test_intentionally_held_layered_slide_is_not_flagged(self):
        frames = [*self.frames[:3], self.frames[2], *self.frames[3:]]
        items = self._check(frames, lambda item: frames[item['id']])
        self.assertFalse(any(item.get('transition_review') or item.get('transition') for item in items))


if __name__ == '__main__':
    unittest.main()
