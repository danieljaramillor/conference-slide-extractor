"""The session file follows the exported slide order and keeps OCR uncertainty."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from slide_dedupe import Source, export
from slide_ocr import read_text
from slide_text import slide_text_record


def grid_slide():
    image = np.full((800, 1200, 3), 255, np.uint8)
    labels = [('Group', 'Response', 'Follow-up'), ('A', '47.8%', '12 months'), ('B', '42.8%', '9 months')]
    for x in (110, 420, 730, 1040):
        cv2.line(image, (x, 210), (x, 450), (45, 45, 45), 2)
    for y in (210, 290, 370, 450):
        cv2.line(image, (110, y), (1040, y), (45, 45, 45), 2)
    for row, values in enumerate(labels):
        for col, value in enumerate(values):
            cv2.putText(image, value, (136+310*col, 260+80*row), cv2.FONT_HERSHEY_SIMPLEX,
                        .85, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(image, 'Slide 14 of 22', (910, 767), cv2.FONT_HERSHEY_SIMPLEX, .75, (30,30,30), 2, cv2.LINE_AA)
    return image


class SessionTextTests(unittest.TestCase):
    def test_recovers_ruled_table_and_explicit_page_from_real_ocr(self):
        image = grid_slide()
        with TemporaryDirectory() as directory:
            ocr = read_text(image, directory)
        result = slide_text_record(2, 'session/slide.png', image, ocr)
        self.assertEqual(result['slide_number'], 2)
        self.assertEqual(result['printed_slide_number'], 14)
        self.assertIn('47.8%', result['text'])
        table, = result['tables']
        self.assertEqual(table['method'], 'ruled_grid')
        self.assertEqual([row[1] for row in table['rows']], ['Response','47.8%','42.8%'])

    def test_reconstructs_repeated_cell_columns_and_leaves_prose_as_text(self):
        boxes = []
        for height, values in ((220, ('Group A','15.5%','18')), (262, ('Group B','-8.2%','6')),
                               (302, ('Group C','7.5%','23')), (344, ('Study ended.','Read the methods.','Result pending.'))):
            for x, value in zip((100, 450, 745), values):
                boxes.append({'text':value,'score':.98,'box':[[x,height],[x+85,height],
                                                             [x+85,height+22],[x,height+22]]})
        ocr = {'status': 'ok', 'text':'\n'.join(box['text'] for box in boxes), 'lines':boxes}
        result = slide_text_record(1, 'test.png', np.full((800,1200,3),255,np.uint8), ocr)
        self.assertEqual(result['text'], ocr['text'])
        table, = result['tables']
        self.assertEqual(table['method'], 'aligned_ocr')
        self.assertEqual(table['rows'][1], ['Group B','-8.2%','6'])
        self.assertIsNone(result['printed_slide_number'])

    def test_unpopulated_lane_between_boxes_is_not_a_table_column(self):
        image = grid_slide()
        for top, bottom in ((210,290),(290,370),(370,450)):
            image[top+3:bottom-3,423:727] = 255
        lines = []
        for row, pair in enumerate((('Group','Follow-up'),('A','12 months'),('B','9 months'))):
            y = 237+80*row
            for x, text in zip((136,756),pair):
                lines.append({'text':text,'score':.98,'box':[[x,y],[x+130,y],[x+130,y+27],[x,y+27]]})
        ocr = {'status':'ok','text':'\n'.join(line['text'] for line in lines),'lines':lines}
        record=slide_text_record(1,'schema.png',image,ocr)
        self.assertFalse(record['tables'])
        self.assertEqual(record['text'],ocr['text'])

    def test_export_rebuilds_records_for_review_order_crop_and_exclusions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root/'captures'
            input_path.mkdir()
            image = grid_slide()
            for index in range(3):
                (input_path/f'{index}.png').write_bytes(cv2.imencode('.png',image)[1].tobytes())
            groups = [{'representative': i,'members':[i],'excluded': i==1} for i in (2,1,0)]
            data = {'groups':groups,'settings':{'ocr':'on'},'images':[
                {'id':i, 'name':f'{i}.png', 'crop':[0,0,1,1], 'ocr':{'status':'ok','text':f'Captured {i}', 'lines':[]}}
                for i in range(3)]}
            source = Source(input_path)
            def recognize(crop, cache):
                return {'status':'ok','text':f'Cropped to {crop.shape[1]}','lines':[]}
            with patch('slide_ocr.read_text', side_effect=recognize):
                export(source,data,root/'reviewed',groups,{'2':[.25,0,1,1]})
            source.close()
            result = [json.loads(line) for line in (root/'reviewed/slides.jsonl').read_text('utf-8').splitlines()]
            self.assertEqual([(row['slide_number'],row['text'],row['source']) for row in result],
                             [(1,'Cropped to 900','2.png'), (2,'Captured 0','0.png')])


if __name__ == '__main__':
    unittest.main()
