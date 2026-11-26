"""OCR follows representative changes into the initial native export."""

import io
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from slide_audit import audit
from slide_dedupe import export


class FinalSelectionOCRTests(unittest.TestCase):
    def test_export_reads_new_representative_after_numeric_split(self):
        images = [
            {'id': index, 'name': f'{index}.png', 'crop': [0, 0, 32, 24],
             'crop_confidence': 1, 'crop_mode': 'full', 'slide_width': 32,
             'slide_height': 24, 'slide_pixels': 768, 'sharpness': sharpness}
            for index, sharpness in enumerate([50, 100, 80])
        ]
        encoded = {
            item['name']: cv2.imencode('.png', np.full((24, 32, 3), item['id'], np.uint8))[1].tobytes()
            for item in images
        }
        source = Mock()
        source.read.side_effect = encoded.__getitem__
        recognized = []

        def recognize(slide, cache):
            index = int(slide[0, 0, 0])
            recognized.append(index)
            message = f'Full text from selected capture number {index}'
            return {'status': 'ok', 'text': message, 'characters': len(message),
                    'mean_confidence': 1.0, 'median_text_height': 12}

        def evidence(a, b, **kwargs):
            self.assertEqual(a['ocr']['status'], 'ok')
            self.assertEqual(b['ocr']['status'], 'ok')
            return {'numeric_conflict': {a['id'], b['id']} == {0, 2},
                    'text_cover_a': 1.0, 'text_cover_b': 1.0, 'heading_similarity': 1.0,
                    'text_words_a': 12, 'text_words_b': 12}

        groups = [{'id': 1, 'representative': 2, 'members': [0, 1, 2], 'builds': 0}]
        decisions = [{'image': 2, 'matched': 0, 'changed': .005, 'reason': 'duplicate'}]
        args = SimpleNamespace(ocr='on', ocr_workers=1)
        with (patch('slide_audit.read_text', side_effect=recognize),
              patch('slide_audit.text_evidence', side_effect=evidence),
              patch('slide_audit.audit_speaker_frames', return_value={}),
              patch('slide_dedupe.compare', side_effect=AssertionError('Verified conflict must block re-merge')),
              redirect_stdout(io.StringIO())):
            groups, _, log = audit(source, images, groups, decisions, [], args, None)
        self.assertEqual(recognized, [0, 2, 1])
        self.assertEqual([group['representative'] for group in groups], [1, 2])
        self.assertEqual(log['ocr_new_representatives'], 1)
        self.assertEqual(log['ocr_errors'], 0)
        self.assertEqual(log['ocr_images'], 3)

        with TemporaryDirectory() as directory:
            output = Path(directory)
            export(source, {'images': images, 'groups': groups}, output)
            for index, capture in enumerate([1, 2], 1):
                self.assertEqual((output / 'text' / f'{index:04d}.txt').read_text(),
                                 f'Full text from selected capture number {capture}')
                self.assertTrue((output / 'text' / f'{index:04d}.json').is_file())
