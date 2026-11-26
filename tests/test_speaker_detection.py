"""Exercise the bundled detector and the safeguards around automatic exclusion."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import cv2
import numpy as np

from slide_audit import audit_speaker_frames
from slide_dedupe import run
from slide_speakers import MODEL, create_detector, has_slide_text, speaker_status
from mixed_fixtures import make_slide
from speaker_fixtures import camera_capture, load_portrait


class SpeakerDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.portrait = load_portrait()
        if cls.portrait is None:
            raise AssertionError('Missing public-domain speaker test image')
        cls.detector = create_detector()

    def image(self, fill=18, portrait=True):
        image = np.full((360, 640, 3), fill, np.uint8)
        if portrait:
            height, width = self.portrait.shape[:2]
            image[70:70+height, 265:265+width] = self.portrait
        return image

    def item(self, image_id=1, *, mode='auto-inset', state='ok', lines=()):
        return {'id': image_id, 'name': str(image_id), 'crop': (0., 0., 1., 1.),
                'crop_mode': mode, 'slide_width': 640, 'slide_height': 360,
                'ocr': {'status': state, 'lines': list(lines), 'characters': sum(len(x['text']) for x in lines)}}

    @staticmethod
    def line(text, y):
        return {'text': text, 'score': .98, 'height': 17,
                'box': [[70, y-8], [270, y-8], [270, y+8], [70, y+8]]}

    def test_stage_with_lower_third_is_camera_footage(self):
        item = self.item(lines=[self.line('Confidential moderator', 321)])
        self.assertFalse(has_slide_text(item))
        self.assertEqual(speaker_status(self.detector, self.image(), item), 'speaker-only')

    def test_headshot_with_slide_heading_stays_in_deck(self):
        item = self.item(lines=[self.line('SPEAKERS', 36)])
        self.assertTrue(has_slide_text(item))
        self.assertIsNone(speaker_status(self.detector, self.image(), item))

    def test_bright_headshot_and_dark_fullframe_are_reviewed(self):
        self.assertEqual(speaker_status(self.detector, self.image(fill=155), self.item()), 'review')
        self.assertEqual(speaker_status(self.detector, self.image(), self.item(mode='full')), 'review')

    def test_manual_crop_and_visible_projection_stay_in_deck(self):
        for mode in ('manual', 'projected-screen'):
            with self.subTest(mode=mode):
                self.assertIsNone(speaker_status(self.detector, self.image(), self.item(mode=mode)))

    def test_without_ocr_and_during_crossfade_require_review(self):
        for item in (self.item(state='error'), {**self.item(), 'crossfade': True}):
            with self.subTest(item=item):
                self.assertEqual(speaker_status(self.detector, self.image(), item), 'review')

    def test_dark_background_without_person_is_kept(self):
        self.assertIsNone(speaker_status(self.detector, self.image(portrait=False), self.item()))

    def test_small_camera_crop_remains_detectable(self):
        clipped = cv2.resize(self.image(), (160, 90), interpolation=cv2.INTER_AREA)
        item = {**self.item(), 'slide_width': 160, 'slide_height': 90}
        self.assertEqual(speaker_status(self.detector, clipped, item), 'speaker-only')

    def test_model_loads_from_unicode_path(self):
        with tempfile.TemporaryDirectory(prefix='extraction-ñ-') as directory:
            model = Path(directory) / 'detector-ñ.onnx'
            model.write_bytes(MODEL.read_bytes())
            with patch('slide_speakers.MODEL', model):
                self.assertEqual(speaker_status(create_detector(), self.image(), self.item()), 'speaker-only')

    def run_audit(self, group, items, ocr_enabled=True):
        source = SimpleNamespace(read=lambda name: self.image())
        return audit_speaker_frames(source, [group], items, SimpleNamespace(), lambda image: image, ocr_enabled)

    def test_isolated_camera_group_is_excluded_with_audit_evidence(self):
        group = {'id': 7, 'representative': 1, 'members': [1]}
        report = self.run_audit(group, {1: self.item()})
        self.assertEqual(report['status'], 'ok')
        self.assertEqual(report['auto_excluded'], [{'group': 7, 'representative': 1}])
        self.assertEqual(group['auto_exclusion'], 'speaker-only')
        self.assertTrue(group['excluded'])

    def test_alternate_with_slide_content_prevents_exclusion(self):
        group = {'id': 7, 'representative': 1, 'members': [1, 2]}
        alternate = self.item(2, lines=[self.line('A short heading', 45)])
        report = self.run_audit(group, {1: self.item(), 2: alternate})
        self.assertEqual(report['auto_excluded'], [])
        self.assertEqual(group['representative'], 2)
        self.assertEqual(group['members'], [1, 2])

    def test_unverified_alternate_requires_group_review(self):
        group = {'id': 7, 'representative': 1, 'members': [1, 2]}
        report = self.run_audit(group, {1: self.item(), 2: self.item(2, state='error')})
        self.assertEqual(report['auto_excluded'], [])
        self.assertEqual(report['needs_review'], [{'group': 7, 'representative': 1}])
        self.assertTrue(group['speaker_review'])

    def test_missing_model_leaves_preserved_candidates_in_review(self):
        group = {'id': 7, 'representative': 1, 'members': [1]}
        with patch('slide_audit.create_detector', side_effect=FileNotFoundError('Model not found')):
            report = self.run_audit(group, {1: self.item()})
        self.assertEqual(report['status'], 'unavailable')
        self.assertEqual(report['auto_excluded'], [])
        self.assertTrue(group['speaker_review'])

    def test_single_frame_detection_error_requires_review(self):
        group = {'id': 7, 'representative': 1, 'members': [1]}
        with patch('slide_audit.speaker_status', side_effect=cv2.error('Inference failed')):
            report = self.run_audit(group, {1: self.item()})
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['auto_excluded'], [])
        self.assertEqual(report['needs_review'], [{'group': 7, 'representative': 1}])
        self.assertTrue(group['speaker_review'])

    def test_disabled_ocr_never_excludes(self):
        group = {'id': 7, 'representative': 1, 'members': [1]}
        report = self.run_audit(group, {1: self.item()}, ocr_enabled=False)
        self.assertFalse(report['auto_exclusion_enabled'])
        self.assertEqual(report['auto_excluded'], [])
        self.assertTrue(group['speaker_review'])

    def test_scan_exports_the_photo_slide_and_excludes_the_camera_capture(self):
        slide = make_slide(title='Presenter profile')
        height, width = self.portrait.shape[:2]
        slide[230:230+height, 1460:1460+width] = self.portrait
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'pool.zip'
            with zipfile.ZipFile(archive, 'w') as zip_archive:
                for name, image in [('camera.png', camera_capture()), ('profile.png', slide)]:
                    ok, encoded = cv2.imencode('.png', image)
                    self.assertTrue(ok)
                    zip_archive.writestr(name, encoded.tobytes())
            output = root / 'results'
            args = argparse.Namespace(input=str(archive), output=str(output), builds='complete', crop=None,
                                      workers=2, ocr='on', ocr_workers=1, ocr_cache=str(root / 'cache'))
            with contextlib.redirect_stdout(io.StringIO()):
                run(args)
            data = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(data['stats']['processed'], 2)
            self.assertEqual(data['stats']['selected'], 1)
            groups = {data['images'][g['representative']]['name']: g for g in data['groups']}
            self.assertEqual(groups['camera.png']['auto_exclusion'], 'speaker-only')
            self.assertFalse(groups['profile.png'].get('excluded', False))
            image = cv2.imdecode(np.frombuffer((output / 'selected/slides/0001.png').read_bytes(), np.uint8),
                                 cv2.IMREAD_COLOR)
            self.assertTrue(np.array_equal(image, slide))


if __name__ == '__main__':
    unittest.main()
