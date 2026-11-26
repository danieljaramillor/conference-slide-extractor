"""Unresolved numeric changes stay separate with inspectable review evidence."""

from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from slide_audit import NUMERIC_REVIEW, audit


class NumericReviewTests(unittest.TestCase):
    def test_unresolved_changes_from_each_merge_path_preserve_both_captures(self):
        for path in ('grouped', 'cross-group', 'local only', 'cross-group local only'):
            with self.subTest(path=path):
                self.check_path(path)

    def check_path(self, path):
        items = [
            {'id': index, 'name': f'{index}.png', 'crop': [0, 0, 1, 1],
             'crop_confidence': 1, 'crop_mode': 'full', 'slide_width': 32,
             'slide_height': 24, 'slide_pixels': 768, 'sharpness': 60, 'sha256': str(index)}
            for index in range(2)
        ]
        samples = [cv2.imencode('.png', np.full((24, 32, 3), i, np.uint8))[1].tobytes() for i in range(2)]
        source = Mock()
        source.read.side_effect = {item['name']: sample for item, sample in zip(items, samples)}.__getitem__
        ocr = {'status': 'ok', 'text': 'Case results for six study sites', 'characters': 40,
               'mean_confidence': .98, 'median_text_height': 14}
        cell = {'bounds_a': [9, 4, 18, 14], 'bounds_b': [9, 4, 18, 14],
                'text_a': '42.8%', 'text_b': '47.8%', 'read_a': '', 'read_b': '47.8%',
                'confidence_a': 0., 'confidence_b': .86}

        def text_evidence(first, second, *, native_pair, reread_numeric):
            return {'numeric_conflict': 'local only' not in path,
                    'numeric_reviews': [cell] if reread_numeric and 'local only' not in path else [],
                    'text_cover_a': 1., 'text_cover_b': 1., 'heading_similarity': 1.,
                    'text_words_a': 15, 'text_words_b': 15}

        def local_evidence(first, second, *, occlusions, unresolved):
            if 'local only' in path:
                unresolved.append(cell)
            return []

        viable = {'duplicate': True, 'content_replacement': False, 'localized_reveal': False,
                  'mae': .001, 'changed': .001, 'coverage_a': .99, 'coverage_b': .99}
        if path.startswith('cross-group'):
            groups = [{'id': index + 1, 'representative': index, 'members': [index], 'builds': 0}
                      for index in range(2)]
            decisions = []
            pair_ids = (0, 1)
        else:
            groups = [{'id': 1, 'representative': 0, 'members': [0, 1], 'builds': 0}]
            decisions = [{'image': 1, 'matched': 0, 'reason': 'duplicate', **viable}]
            pair_ids = (1, 0)
        with (patch('slide_audit.read_text', return_value=ocr),
              patch('slide_audit.text_evidence', side_effect=text_evidence),
              patch('slide_audit.local_numeric_evidence', side_effect=local_evidence),
              patch('slide_audit.audit_speaker_frames', return_value={}),
              patch('slide_dedupe.compare', return_value=viable),
              redirect_stdout(io.StringIO())):
            result, suggestions, log = audit(source, items, groups, decisions, [{'a': 0, 'b': 1}],
                                             SimpleNamespace(ocr='on', ocr_workers=1), None)
        self.assertEqual([group['members'] for group in result], [[0], [1]])
        self.assertEqual(log['numeric_reviews'], [dict(a=pair_ids[0], b=pair_ids[1], cells=[cell], group=1)])
        self.assertEqual(log['numeric_splits'], [])
        self.assertEqual(suggestions, [])
        self.assertIn(NUMERIC_REVIEW, result[0]['review_reasons'])
        self.assertNotIn(NUMERIC_REVIEW, result[1]['review_reasons'])


if __name__ == '__main__':
    unittest.main()
