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


def grid_ocr(response_box):
    lines = []
    for row, values in enumerate((('Group','Response','Follow-up'),('A','47.8%','12 months'),('B','42.8%','9 months'))):
        for column, label in enumerate(values):
            x, y = 136+310*column, 232+80*row
            corners = [[x,y], [x+132,y], [x+132,y+31], [x,y+31]]
            lines.append({'text': label, 'score': .98, 'box': response_box if (row,column)==(1,1) else corners})
    return {'status': 'ok', 'text': '\n'.join(line['text'] for line in lines), 'lines': lines}


class SessionTextTests(unittest.TestCase):
    def test_export_adds_verified_risk_strip_and_keeps_slide_text(self):
        image = grid_slide()
        ocr = grid_ocr([[446,312], [578,312], [578,343], [446,343]])
        other = {'bounds': [60, 50, 280, 130], 'rows': [['Group'], ['A']]}
        old_strip = {'bounds': [240, 540, 720, 580], 'rows': [['Events'], ['018']]}
        risk_strip = {'method': 'risk_count_strip', 'bounds': [220, 520, 760, 600],
                      'rows': [['Months', '0', '3'], ['At risk', '18', '17'], ['Events', '0', '1']]}
        with (patch('slide_text._ruled_tables', return_value=([other, old_strip], [])),
              patch('slide_text.aligned_tables', return_value=[]),
              patch('slide_text.risk_count_table', return_value=risk_strip)):
            record = slide_text_record(1, 'survival.png', image, ocr)
        self.assertEqual(record['text'], ocr['text'])
        self.assertEqual(record['tables'], [other, risk_strip])

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

    def test_box_crossing_rows_uses_separate_native_crops(self):
        ocr = grid_ocr([[446,332], [578,332], [578,416], [446,416]])
        result = slide_text_record(1, 'table.png', grid_slide(), ocr)
        table, = result['tables']
        self.assertEqual([row[1] for row in table['rows']], ['Response','47.8%','42.8%'])
        self.assertEqual(result['text'], ocr['text'])
        self.assertFalse(table['review_cells'])
        self.assertEqual([(cell['row'],cell['column'],cell['bounds']) for cell in table['cell_sources']],
                         [(2,2,[420,290,730,370]), (3,2,[420,370,730,450])])

    def test_unavailable_reread_flags_cropped_cells_without_assigning_their_text(self):
        ocr = grid_ocr([[446,312], [838,312], [838,343], [446,343]])
        with patch('slide_ocr._recognize_cell', side_effect=RuntimeError('Model unavailable')):
            result = slide_text_record(1, 'table.png', grid_slide(), ocr)
        table, = result['tables']
        self.assertEqual(table['rows'][1], ['A',None,None])
        self.assertEqual(table['rows'][2], ['B','42.8%','9 months'])
        self.assertEqual(table['status'], 'partial')
        self.assertFalse(table['cell_sources'])
        self.assertEqual(table['review_cells'], [
            {'row':2,'column':2,'bounds':[420,290,730,370],'reason':'crosses_cell_boundary'},
            {'row':2,'column':3,'bounds':[730,290,1040,370],'reason':'crosses_cell_boundary'},
        ])

    def test_ink_touching_a_grid_line_prevents_native_reread(self):
        image = grid_slide()
        cv2.line(image, (540,347), (540,402), (0,0,0), 2)
        ocr = grid_ocr([[446,332], [578,332], [578,416], [446,416]])
        with patch('slide_ocr._recognize_cell', return_value=('99%', .99)) as recognizer:
            table, = slide_text_record(1, 'table.png', image, ocr)['tables']
        recognizer.assert_not_called()
        self.assertEqual([row[1] for row in table['rows']], ['Response',None,None])
        self.assertEqual([(cell['row'],cell['column']) for cell in table['review_cells']], [(2,2), (3,2)])

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

    def test_keeps_single_value_column_and_header_separate_from_chart_axis(self):
        def box(text, x, y, width):
            return {'text':text,'score':.98,'box':[[x,y],[x+width,y],[x+width,y+25],[x,y+25]]}

        lines = [box('Participant',100,175,145),box('Result',447,175,75)]
        for row,label,value in zip(range(6),('Arm A','Arm B','Arm C','Arm D','Arm E','Arm F'),
                                   ('22 (36%)','12 (32%)','9 (15%)','0','3 (5%)','2 (3%)')):
            y = 220+40*row
            lines.extend([box(label,100,y,80),box(value,450,y,90),box(str(20-10*row),700,y,38),
                          box('• Adjacent text',845,y,170)])
        image = np.full((800,1200,3),255,np.uint8)
        ocr = {'status':'ok','text':'\n'.join(line['text'] for line in lines),'lines':lines}
        table, = slide_text_record(1,'table.png',image,ocr)['tables']
        self.assertEqual(table['rows'][0],['Participant','Result'])
        self.assertEqual(table['rows'][3],['Arm C','9 (15%)'])
        self.assertEqual(table['cell_bounds'][3][1],[450,300,540,325])
        self.assertEqual(len(table['rows']),7)

    def test_keeps_blank_numeric_lane_in_row_without_shifting_later_value(self):
        lines = []
        for row, fields in enumerate((('A','17.5%','25'),('B',None,'18'),('C','8.0%','6'),('D','9.1%','9'))):
            y = 220+40*row
            for column, value in enumerate(fields):
                if value is None:
                    continue
                x = 100+275*column
                lines.append({'text': value, 'score':.98,
                              'box':[[x,y],[x+60,y],[x+60,y+24],[x,y+24]]})
        image = np.full((750,1100,3),255,np.uint8)
        ocr = {'status':'ok','text':'\n'.join(line['text'] for line in lines),'lines':lines}
        table, = slide_text_record(1,'table.png',image,ocr)['tables']
        self.assertEqual(table['rows'][1],['B',None,'18'])
        self.assertIsNone(table['cell_bounds'][1][1])

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

    def test_declined_grid_cannot_supply_its_boxes_to_aligned_fallback(self):
        lines = []
        for row, label in enumerate(('A', 'B', 'C')):
            y = 239+80*row
            for x, value in zip((145, 447, 600), (label, '12', '15')):
                lines.append({'text': value, 'score': .98,
                              'box': [[x,y],[x+36,y],[x+36,y+20],[x,y+20]]})
        ocr = {'status': 'ok', 'text': '\n'.join(line['text'] for line in lines), 'lines': lines}
        record = slide_text_record(1, 'declined-grid.png', grid_slide(), ocr)
        self.assertEqual(record['tables'], [])
        self.assertEqual(record['text'], ocr['text'])

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
