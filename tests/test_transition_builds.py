"""Exercise real scans so transition suppression cannot discard clean builds."""

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

from mixed_fixtures import embed, make_slide
import slide_dedupe
from transition_fixtures import build_transition_sequence, slide_build_stages, timestamped


class TransitionBuildTests(unittest.TestCase):
    def scan(self, captures, *, ocr='off', builds='complete'):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        archive = root / 'captures.zip'
        with zipfile.ZipFile(archive, 'w') as zipped:
            for name, image in captures:
                zipped.writestr(name, cv2.imencode('.png', image)[1].tobytes())

        output = root / 'result'
        args = argparse.Namespace(input=archive, output=output, workers=2, crop=None,
                                  builds=builds, ocr=ocr, ocr_workers=1, ocr_cache=root / 'cache')
        with contextlib.redirect_stdout(io.StringIO()):
            slide_dedupe.run(args)
        manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        self.assertCountEqual([image for group in manifest['groups'] for image in group['members']],
                              range(len(captures)))
        return manifest, output

    def assert_exported_frames(self, manifest, output, captures, expected_ids):
        included_ids = [group['representative'] for group in manifest['groups'] if not group.get('excluded')]
        self.assertEqual(included_ids, expected_ids)
        slides = sorted((output / 'selected' / 'slides').glob('*.png'))
        self.assertEqual(len(slides), len(expected_ids))
        for path, image_id in zip(slides, expected_ids):
            pixels = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(pixels, captures[image_id][1])

    def test_group_keeps_late_panel_and_qr_reveals_across_crop_layouts(self):
        base, panel, complete = slide_build_stages()
        faint_panel = cv2.addWeighted(base, .5, panel, .5, 0)
        # The crop contains the exact middle stage pixels, with nearby UI and a speaker pane.
        with_margin = embed(panel, (1300, 760), (42, 108, 960, 540), (174, 120, 190))
        cv2.rectangle(with_margin, (1020, 108), (1280, 330), (25, 26, 27), -1)
        captures = timestamped((base, faint_panel, with_margin, complete))
        manifest, output = self.scan(captures)
        self.assertEqual(manifest['images'][2]['crop_mode'], 'auto-inset')
        self.assertEqual(len(manifest['groups']), 1)
        self.assert_exported_frames(manifest, output, captures, [3])

    def test_broad_blend_between_slides_does_not_replace_a_completed_build(self):
        captures = build_transition_sequence()
        for build_policy, expected in (('complete', [2, 4]), ('all', [0, 1, 2, 4])):
            with self.subTest(build_policy=build_policy):
                manifest, output = self.scan(captures, builds=build_policy)
                self.assert_exported_frames(manifest, output, captures, expected)

    def test_consecutive_blends_preserve_only_the_clean_slides(self):
        base, panel, complete, _, next_slide, repeated = (
            image for _, image in build_transition_sequence())
        blends = [cv2.addWeighted(complete, 1 - weight, next_slide, weight, 0)
                  for weight in (.20, .40, .60, .80)]
        captures = timestamped((base, panel, complete, *blends, next_slide, repeated))
        manifest, output = self.scan(captures)
        self.assert_exported_frames(manifest, output, captures, [2, 7])
        excluded = [group['representative'] for group in manifest['groups'] if group.get('excluded')]
        self.assertCountEqual(excluded, [3, 4, 5, 6])

    def test_consecutive_solid_reveals_are_preserved_by_the_selected_build_policy(self):
        base, panel, complete = slide_build_stages()
        final = complete.copy()
        cv2.putText(final, 'Week 24 chart', (600, 147),
                    cv2.FONT_HERSHEY_SIMPLEX, .75, (80, 30, 90), 2)
        cv2.rectangle(final, (610, 165), (816, 245), (80, 30, 90), 2)
        for x, top in ((630, 215), (690, 200), (750, 179)):
            cv2.rectangle(final, (x, top), (x + 32, 230), (150, 80, 25), -1)
        captures = timestamped((base, panel, complete, final))
        for build_policy, expected in (('complete', [3]), ('all', [0, 1, 2, 3])):
            with self.subTest(build_policy=build_policy):
                manifest, output = self.scan(captures, builds=build_policy)
                self.assert_exported_frames(manifest, output, captures, expected)
                self.assertFalse(any(group.get('excluded') for group in manifest['groups']))

    def test_replacement_chart_with_changed_value_stays_separate_from_build(self):
        base = make_slide(title='Treatment study results', number='47.8')
        complete = base.copy()
        changed = make_slide(title='Treatment study results', number='42.8')
        for image, bars in ((complete, (86, 164, 260, 285)), (changed, (300, 228, 134, 65))):
            for index, height in enumerate(bars):
                x = 1200 + index * 104
                cv2.rectangle(image, (x, 740 - height), (x + 70, 740), (80, 20, 120), -1)
            cv2.putText(image, 'Long term response', (1130, 350),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (90, 30, 20), 3)

        captures = timestamped((base, complete, changed))
        manifest, output = self.scan(captures, ocr='on')
        self.assert_exported_frames(manifest, output, captures, [1, 2])
        for index, value in ((1, '47.8'), (2, '42.8')):
            text = (output / 'selected' / 'text' / f'{index:04}.txt').read_text(encoding='utf-8')
            self.assertIn(f'Study result: {value} percent', text)

    def test_ocr_equivalent_opaque_build_beats_faded_text_on_dark_slide(self):
        base = cv2.bitwise_not(make_slide(title='Slide build outcomes'))
        complete = base.copy()
        cv2.putText(complete, 'Long term analysis now added', (1130, 605),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.38, (230, 230, 220), 3, cv2.LINE_AA)
        cv2.putText(complete, 'Observed benefits for both cohorts', (1120, 680),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.15, (240, 240, 230), 3, cv2.LINE_AA)
        faint = cv2.addWeighted(base, .6, complete, .4, 0)
        captures = timestamped((base, faint, complete))
        manifest, output = self.scan(captures, ocr='on')
        self.assertEqual(len(manifest['groups']), 1)
        self.assert_exported_frames(manifest, output, captures, [2])
        text = (output / 'selected' / 'text' / '0001.txt').read_text(encoding='utf-8')
        self.assertIn('Long term analysis now added', text)
        self.assertIn('Observed benefits for both cohorts', text)


if __name__ == '__main__':
    unittest.main()
