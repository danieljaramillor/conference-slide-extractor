import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from slide_dedupe import Source, export
from slide_pages import confirm_footer_numbers, footer_number


def footer_ocr(value, x=963, y=939):
    text = str(value)
    box = [[x, y], [x + 15, y], [x + 15, y + 25], [x, y + 25]]
    return {'status': 'ok', 'text': text, 'lines': [{'text': text, 'score': .99, 'box': box}]}


class ExportTextTests(unittest.TestCase):
    def test_footer_numbers_need_stable_consecutive_observations(self):
        records = [{'printed_slide_number': None} for _ in range(6)]
        ocr = [footer_ocr(3), footer_ocr(4), footer_ocr(5), footer_ocr(6, y=859),
               footer_ocr(7), footer_ocr(8)]
        confirm_footer_numbers(records, [footer_number(item, 1000, 1000) for item in ocr])
        self.assertEqual([row['printed_slide_number'] for row in records], [3, 4, 5, None, None, None])

    def test_explicit_number_can_prevent_a_conflicting_footer_run(self):
        ocr = [footer_ocr(value) for value in (3, 4, 5)]
        records = [{'printed_slide_number': value} for value in (None, 17, None)]
        confirm_footer_numbers(records, [footer_number(item, 1000, 1000) for item in ocr])
        self.assertEqual([row['printed_slide_number'] for row in records], [None, 17, None])

    def test_export_refinement_does_not_replace_grouping_ocr(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            captures = root / 'captures'
            captures.mkdir()
            photo = np.full((100, 160, 3), 255, np.uint8)
            (captures / 'slide.png').write_bytes(cv2.imencode('.png', photo)[1].tobytes())
            baseline = {'status': 'ok', 'text': 'Baseline read', 'lines': []}
            refined = {**baseline, 'text': 'Baseline read with 47.8%',
                       'export_refinement': {'applied': True}}
            item = {'id': 0, 'name': 'slide.png', 'crop': [0, 0, 1, 1], 'ocr': baseline}
            manifest = {'images': [item], 'groups': [{'representative': 0, 'members': [0]}],
                        'settings': {'ocr': 'on', 'ocr_workers': 2, 'ocr_cache': str(root / 'cache')}}
            selected = root / 'selected'
            source = Source(captures)
            try:
                with patch('slide_ocr.refine_export_text', return_value=refined) as refine:
                    self.assertEqual(export(source, manifest, selected), {'attempted': 1, 'applied': 1})
                    self.assertEqual(refine.call_count, 1)
            finally:
                source.close()
            self.assertEqual(item['ocr'], baseline)
            self.assertEqual((selected / 'text/0001.txt').read_text(encoding='utf-8'), refined['text'])
            record = json.loads((selected / 'slides.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(record['text'], refined['text'])
