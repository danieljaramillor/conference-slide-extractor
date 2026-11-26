"""Automatic transition detection keeps evidence and requires reliable neighbors."""
import unittest

import cv2
import numpy as np

from slide_transitions import mark_crossfades


def _frame(seed):
    rng = np.random.default_rng(seed)
    background, ink = (220, 25) if seed % 2 else (35, 235)
    frame = np.full((270, 480), background, dtype=np.uint8)
    for y in range(16, 260, 23):
        for x in range(14, 470, 100):
            size = int(rng.integers(6, 14))
            offset = int(rng.integers(0, 12))
            cv2.rectangle(frame, (x + offset, y), (x + offset + size, y + size), ink, 2)
            cv2.putText(frame, str(rng.integers(1000, 10000)), (x + 22, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, .42, ink, 1, cv2.LINE_AA)
    return cv2.GaussianBlur(frame, (3, 3), 0)


def _captures(frames):
    return [{'gray': image, 'id': index, 'name': f'session/frame_{index + 1:02}.jpg'}
            for index, image in enumerate(frames)]


class TransitionTests(unittest.TestCase):
    def setUp(self):
        self.before, self.after = _frame(5), _frame(6)
        self.middle = cv2.addWeighted(self.before, .36, self.after, .64, 0)

    def test_equal_size_overlap_records_both_source_frames(self):
        frames = _captures((self.before, self.middle, self.after))
        self.assertEqual([item['id'] for item in mark_crossfades(frames)], [1])
        evidence = frames[1]['transition']
        self.assertEqual((evidence['kind'], evidence['before'], evidence['after']), ('crossfade', 0, 2))
        self.assertAlmostEqual(evidence['weight'], .64, delta=.01)

    def test_crop_layout_change_registers_the_smaller_pane(self):
        full_capture = np.full_like(self.after, 245)
        full_capture[30:206, 14:326] = cv2.resize(self.after, (312, 176))
        cv2.putText(full_capture, 'POLL', (360, 135), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 30, 3)
        frames = _captures((self.before, self.middle, full_capture))
        for capture in frames[:2]:
            capture.update(crop=(.06, .12, .8, .88), crop_mode='auto-inset')
        frames[-1].update(crop=(0, 0, 1, 1), crop_mode='full')
        self.assertEqual([item['id'] for item in mark_crossfades(frames)], [1])
        self.assertEqual(frames[1]['transition']['kind'], 'resized-crossfade')

    def test_missing_unrelated_or_projected_neighbor_keeps_capture(self):
        for change in ({'id': 3}, {'name': 'next-session/frame_03.jpg'},
                       {'name': 'session/frame_09.jpg'}, {'crop_mode': 'projected-screen'}):
            with self.subTest(change=change):
                frames = _captures((self.before, self.middle, self.after))
                frames[-1].update(change)
                self.assertEqual(mark_crossfades(frames), [])
                self.assertNotIn('transition', frames[1])

    def test_dissolve_run_reuses_both_clear_endpoints(self):
        intermediate = [cv2.addWeighted(self.before, 1 - alpha, self.after, alpha, 0)
                        for alpha in (.20, .40, .60, .80)]
        frames = _captures((self.before, *intermediate, self.after))
        transitions = mark_crossfades(frames)
        self.assertEqual([item['id'] for item in transitions], [1, 2, 3, 4])
        self.assertEqual({(item['transition']['before'], item['transition']['after'])
                          for item in transitions}, {(0, 5)})

    def test_exact_opacity_reveal_gets_soft_preference_only(self):
        earlier = np.full((270, 480), 235, dtype=np.uint8)
        cv2.putText(earlier, 'Title and original result', (32, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, .78, 20, 2)
        complete = earlier.copy()
        cv2.putText(complete, 'Additional result', (130, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, 20, 2)
        intermediate = cv2.addWeighted(earlier, .60, complete, .40, 0)
        frames = _captures((earlier, intermediate, complete))
        self.assertEqual(mark_crossfades(frames), [])
        self.assertTrue(frames[1]['crossfade'])
        self.assertNotIn('crossfade', frames[2])


if __name__ == '__main__':
    unittest.main()
