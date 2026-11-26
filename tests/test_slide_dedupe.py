import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import cv2
import numpy as np

import slide_dedupe as sd


def slide(title='Trial results', extra=False, dark=False):
    im = np.full((540, 960, 3), 35 if dark else 248, np.uint8)
    ink = (235, 235, 235) if dark else (100, 45, 10)
    cv2.putText(im, title, (40, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.3, ink, 2)
    cv2.putText(im, 'Treatment outcomes: cohort A', (40, 150), cv2.FONT_HERSHEY_SIMPLEX, .8, ink, 2)
    cv2.rectangle(im, (50, 210), (400, 440), (150, 80, 25), 3)
    cv2.line(im, (70, 420), (370, 240), ink, 3)
    if extra:
        cv2.putText(im, 'Additional results', (510, 180), cv2.FONT_HERSHEY_SIMPLEX, .8, ink, 2)
        cv2.rectangle(im, (530, 220), (850, 440), (80, 20, 130), 3)
    return im


def feat(im):
    _, gray, edges, _ = sd.features(im)
    return {'gray': gray, 'edges': edges}


def broadcast(im, speaker):
    frame = np.full((1080, 1920, 3), (174, 120, 190), np.uint8)
    frame[154:958, 16:1444] = cv2.resize(im, (1428, 804))
    cv2.rectangle(frame, (1460, 154), (1900, 480), speaker, -1)
    return frame


class MatchingTests(unittest.TestCase):
    def test_duplicate_compression(self):
        a = slide()
        b = cv2.imdecode(cv2.imencode('.jpg', a, [cv2.IMWRITE_JPEG_QUALITY, 80])[1], 1)
        self.assertTrue(sd.compare(feat(a), feat(b))['duplicate'])

    def test_build_contains_earlier_content(self):
        metric = sd.compare(feat(slide()), feat(slide(extra=True)))
        self.assertTrue(metric['build'], metric)
        self.assertFalse(metric['duplicate'])

    def test_small_panel_reveal_beats_an_equally_sharp_earlier_capture(self):
        early = slide()
        for index in range(8):
            cv2.putText(early, f'Cohort {index}: baseline outcome for patients',
                        (50, 192 + index * 36), cv2.FONT_HERSHEY_SIMPLEX, .65, (90, 30, 10), 2)
        complete = early.copy()
        cv2.putText(complete, '47.8%', (658, 322), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 2)
        cv2.rectangle(complete, (646, 286), (774, 344), (80, 30, 90), 3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'slides.zip'
            with zipfile.ZipFile(archive, 'w') as images:
                images.writestr('01.png', cv2.imencode('.png', early)[1].tobytes())
                images.writestr('02.png', cv2.imencode('.png', complete)[1].tobytes())
            for policy, expected in [('complete', [1]), ('all', [0, 1])]:
                args = argparse.Namespace(input=str(archive), output=str(root / policy), crop=None,
                                          workers=2, builds=policy)
                with contextlib.redirect_stdout(io.StringIO()):
                    sd.run(args)
                manifest = json.loads((root / policy / 'manifest.json').read_text())
                self.assertEqual([group['representative'] for group in manifest['groups']], expected)

    def test_different_titles_on_shared_background_do_not_merge(self):
        for dark in (True, False):
            metric = sd.compare(feat(slide('Introduction', dark=dark)), feat(slide('Clinical foundations', dark=dark)))
            self.assertFalse(metric['duplicate'], metric)
            self.assertFalse(metric['build'], metric)

    def test_speaker_motion_is_outside_crop(self):
        a, b = broadcast(slide(), (0, 0, 0)), broadcast(slide(), (20, 230, 30))
        box, mode = sd.detect_crop(a)
        self.assertEqual(mode, 'auto-inset')
        self.assertAlmostEqual(box[0], 16/1920, delta=.01)
        self.assertAlmostEqual(box[2], 1444/1920, delta=.01)
        self.assertTrue(sd.compare(feat(sd.crop_image(a, box)), feat(sd.crop_image(b, box)))['duplicate'])

    def test_full_slide_is_not_inset(self):
        self.assertEqual(sd.detect_crop(slide())[1], 'full')

    def test_natural_order(self):
        self.assertEqual(sorted(['image_10.jpg','image_2.jpg','image_1.jpg'],key=sd.natural_key), ['image_1.jpg','image_2.jpg','image_10.jpg'])

    def test_prefers_clean_frame_to_crossfade(self):
        before = np.full((270, 480), 220, dtype=np.uint8)
        after = before.copy()
        cv2.rectangle(after, (90, 45), (340, 180), 20, -1)
        transition = cv2.addWeighted(before, .6, after, .4, 0)
        frames = [{'name': f'session/image_00-00-{second:02}.jpg', 'gray': gray,
                   'slide_pixels': 1_000_000, 'sharpness': sharpness}
                  for second, gray, sharpness in [(0, before, 90), (4, transition, 400), (8, after, 90)]]
        sd.mark_crossfades(frames)
        self.assertTrue(frames[1]['crossfade'])
        self.assertFalse(sd.better_frame(frames[1], frames[0], {'duplicate': True}))

    def test_zip_end_to_end_and_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'images.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                for name, im in [('1.png',slide()),('2.png',slide()),('3.png',slide(extra=True)),('4.png',slide('Different trial'))]:
                    z.writestr('nested/'+name, cv2.imencode('.png',im)[1].tobytes())
                z.writestr('nested/broken.jpg',b'not an image')
                z.writestr('../ignored.txt',b'not extracted')
            args = argparse.Namespace(input=str(archive),output=str(root/'result'),crop=None,workers=2,builds='complete')
            with contextlib.redirect_stdout(io.StringIO()):
                sd.run(args)
            m = json.loads((root/'result/manifest.json').read_text())
            self.assertEqual(m['stats']['processed'],4)
            self.assertEqual(len(m['errors']),1)
            self.assertEqual(len(m['groups']),2)
            built = next(g for g in m['groups'] if len(g['members'])==3)
            self.assertEqual(built['representative'],2)
            self.assertEqual(len(list((root/'result/selected/slides').glob('*.png'))),2)
            self.assertIn('Conference slide review',(root/'result/review.html').read_text())
            built['excluded']=True
            src=sd.Source(archive)
            sd.export(src,m,root/'reviewed',m['groups'])
            src.close()
            self.assertEqual(len(list((root/'reviewed/slides').glob('*.png'))),1)
            self.assertFalse((root.parent/'ignored.txt').exists())


if __name__ == '__main__':
    unittest.main()
