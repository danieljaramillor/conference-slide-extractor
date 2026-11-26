"""Exercise selective export OCR without changing scan-time evidence."""

from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from slide_ocr import ENGINE_VERSION, read_text, refine_export_text


def _line(text, y, *, height=20, score=.99):
    return {'text': text, 'score': score, 'height': height,
            'box': [[50, y], [190, y], [190, y + height], [50, y + height]]}


def _ocr(lines):
    return {'status': 'ok', 'engine': ENGINE_VERSION, 'lines': lines,
            'text': '\n'.join(line['text'] for line in lines)}


class HighResolutionOCRTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((800, 1440, 3), dtype=np.uint8)

    def test_keeps_simple_or_small_native_exports_without_reread(self):
        simple = _ocr([_line('Conference title', 200), _line('2030', 710)])
        dense = _ocr([_line(f'Label {n}', 30 * n) for n in range(8)])
        with patch('slide_ocr._read_text') as read:
            self.assertIs(refine_export_text(self.image, simple, 'unused'), simple)
            self.assertIs(refine_export_text(self.image[:400, :800], dense, 'unused'), dense)
            read.assert_not_called()

    def test_adds_missing_labels_and_keeps_conflicting_values(self):
        source = [_line(label, 40 * n) for n, label in enumerate(
            ('-5%', '25.6%', '10', 'SitesAcross', 'First trial', 'Follow-up', 'Controls', 'Patients'), 1)]
        original = _ocr(source)
        higher = _ocr([_line('5 %', 40), _line('25. 6%', 80), _line('1 0', 120),
                       _line('Sites Across', 160), _line('32%', 205), _line('garble', 365, score=.77)])
        with patch('slide_ocr._read_text', return_value=higher):
            result = refine_export_text(self.image, original, 'unused')
        self.assertEqual([line['text'] for line in result['lines']],
                         ['-5%', '25.6%', '10', 'Sites Across', 'First trial', 'Follow-up', 'Controls', 'Patients'])
        self.assertEqual(result['export_refinement']['respaced_lines'], 1)
        self.assertEqual(source[3]['text'], 'SitesAcross')

    def test_new_labels_follow_native_positions(self):
        original = _ocr([_line(f'Row {n}', 70 * n) for n in range(8)])
        higher = _ocr([_line('−22%', 40), _line('12', 395, score=.85)])
        with patch('slide_ocr._read_text', return_value=higher):
            result = refine_export_text(self.image, original, 'unused')
        self.assertEqual(result['export_refinement']['added_lines'], 1)
        self.assertEqual(result['text'].splitlines()[:3], ['Row 0', '−22%', 'Row 1'])

    def test_caches_standard_and_higher_resolution_reads_separately(self):
        base = [_line(f'Row {n}', 70 * n) for n in range(8)]
        calls = []

        def engine(*, high_res=False):
            calls.append(high_res)
            lines = [*base, _line('−22%', 40)] if high_res else base
            return lambda *_args, **_kwargs: SimpleNamespace(
                txts=[line['text'] for line in lines], scores=[line['score'] for line in lines],
                boxes=[line['box'] for line in lines])

        with TemporaryDirectory() as cache, patch('slide_ocr.engine', side_effect=engine):
            standard = read_text(self.image, cache)
            alternate = refine_export_text(self.image, standard, cache)
            self.assertEqual(read_text(self.image, cache), standard)
            self.assertEqual(refine_export_text(self.image, standard, cache), alternate)
        self.assertEqual(calls, [False, True])
        self.assertNotIn('−22%', standard['text'])
        self.assertIn('−22%', alternate['text'])

    def test_reread_failure_preserves_the_original_fields(self):
        original = _ocr([_line(f'Row {n}', 70 * n) for n in range(8)])
        with patch('slide_ocr._read_text', side_effect=RuntimeError('Model failed')):
            result = refine_export_text(self.image, original, 'unused')
        self.assertEqual(result['text'], original['text'])
        self.assertEqual(result['export_refinement']['status'], 'unavailable')
        self.assertFalse(result['export_refinement']['applied'])


if __name__ == '__main__':
    unittest.main()
