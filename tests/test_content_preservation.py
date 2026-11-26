import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import cv2
from content_fixtures import INK, heading_build_stages, shared_slide, text
import slide_dedupe as sd


class ContentPreservationTests(unittest.TestCase):
    def scan(self, frames):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'captures.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                for index, image in enumerate(frames):
                    bundle.writestr(f'capture-{index}.jpg', cv2.imencode('.jpg', image)[1].tobytes())
            args = argparse.Namespace(input=archive, output=root / 'out', workers=2,
                                      crop=(0, 0, 1, 1), builds='complete', ocr='off')
            with contextlib.redirect_stdout(io.StringIO()):
                sd.run(args)
            manifest = json.loads((args.output / 'manifest.json').read_text(encoding='utf-8'))
        return {group['representative']: group['members'] for group in manifest['groups']}

    def test_changed_heading_survives_a_later_additive_diagram(self):
        earlier, newer, complete = heading_build_stages()
        self.assertTrue(sd.compare(sd.slide_features(earlier), sd.slide_features(newer))['duplicate'])
        groups = self.scan([earlier, earlier, newer, newer, complete, complete])
        self.assertEqual(groups, {0: [0, 1], 4: [2, 3, 4, 5]})

    def test_added_panels_do_not_erase_an_earlier_callout(self):
        earlier = shared_slide()
        newer = earlier.copy()
        for row, value in enumerate(('Assignment to three arms', 'Randomized after induction', 'Repeat evaluation at week eight')):
            text(earlier, value, (570, 146 + row * 19), .57)
        for row, value in enumerate(('Independent arm treatment', 'Open follow-up on treatment', 'Evaluate response after progression')):
            text(newer, value, (570, 146 + row * 19), .57)
        for row in range(3):
            text(newer, 'Treatment group summary', (570, 286 + row * 19), .56)
        complete = newer.copy()
        cv2.rectangle(complete, (560, 367), (900, 473), INK, 2)
        text(complete, 'Trial comparison and follow-up', (570, 421), .65)
        self.assertTrue(sd.compare(sd.slide_features(earlier), sd.slide_features(newer))['build'])
        self.assertEqual(self.scan([earlier, newer, complete]), {0: [0], 2: [1, 2]})

    def test_pointer_mark_does_not_become_a_separate_slide(self):
        first = shared_slide()
        second = first.copy()
        cv2.arrowedLine(second, (424, 224), (415, 215), INK, 1, tipLength=.5)
        cv2.arrowedLine(first, (775, 370), (766, 361), INK, 1, tipLength=.5)
        self.assertEqual(list(self.scan([first, second]).values()), [[0, 1]])

    def test_solid_callout_that_covers_table_cells_stays_separate(self):
        table = shared_slide()
        for row in range(8):
            y = 195 + 17 * row
            fill = (152, 212, 232) if row % 2 else (211, 216, 191)
            cv2.rectangle(table, (556, y), (887, y + 16), fill, -1)
            text(table, 'Primary response | 42.8% | 63.4%', (567, y + 13), .46)
        annotated = table.copy()
        cv2.rectangle(annotated, (592, 221), (882, 308), (65, 94, 47), -1)
        for label, origin in (('Confirmatory assessment', (607, 247)), ('At the next study visit', (607, 271)),
                              ('By planned response schedule', (607, 294))):
            cv2.putText(annotated, label, origin, cv2.FONT_HERSHEY_SIMPLEX, .55, (245, 249, 248), 1, cv2.LINE_AA)
        self.assertEqual(self.scan([table, table, annotated, annotated]), {0: [0, 1], 2: [2, 3]})


if __name__ == '__main__':
    unittest.main()
