import unittest
from unittest.mock import patch

import cv2
import numpy as np

from slide_risk import risk_count_table


def survival_chart(*, zero_tick=True, omit_count=False):
    image = np.full((650, 1000, 3), 255, np.uint8)
    ink = (25, 25, 25)
    cv2.line(image, (225, 397), (675, 397), ink, 1)
    cv2.line(image, (225, 165), (225, 397), ink, 1)
    cv2.putText(image, 'Months', (414, 439), cv2.FONT_HERSHEY_SIMPLEX, .6, ink, 1, cv2.LINE_AA)
    boxes = [('Months', 410., 418., 493., 445., .99)]
    for col, tick in enumerate(('0','3','6','9','12','15')):
        x = 225 + 90*col
        if col or zero_tick:
            cv2.putText(image, tick, (x-4, 414), cv2.FONT_HERSHEY_SIMPLEX, .4, ink, 1, cv2.LINE_AA)
        if col:
            boxes.append((tick, float(x-11), 399., float(x+11), 421., .99))

    rows = [('At risk', '35','26','18','12','5','0'),
            ('Censored', '0','1','6','11','13','16'),
            ('Events', '0','8','11','12','17','19')]
    for row, (label, *values) in enumerate(rows):
        y = 484 + 14*row
        cv2.putText(image, label, (64, y), cv2.FONT_HERSHEY_SIMPLEX, .40, ink, 1, cv2.LINE_AA)
        if label == 'Events':
            boxes.append((label, 63., float(y-14), 128., float(y+4), .99))
        for col, value in enumerate(values):
            if not (omit_count and row == 1 and col == 3):
                cv2.putText(image, value, (220+90*col-6*len(value), y), cv2.FONT_HERSHEY_SIMPLEX,
                            .40, ink, 1, cv2.LINE_AA)
    return image, boxes


class RiskCountTests(unittest.TestCase):
    def test_reads_different_counts_by_row_and_column(self):
        image, boxes = survival_chart()
        table = risk_count_table(image, boxes)
        self.assertEqual(table['method'], 'risk_count_strip')
        self.assertEqual(table['rows'], [['Months','0','3','6','9','12','15'],
                                         ['At risk','35','26','18','12','5','0'],
                                         ['Censored','0','1','6','11','13','16'],
                                         ['Events','0','8','11','12','17','19']])
        self.assertFalse(table['review_cells'])
        cell = table['cell_bounds'][2][3]
        self.assertLess(cell[0], 405)
        self.assertGreater(cell[2], 405)
        self.assertLess(cell[1], 493)
        self.assertGreater(cell[3], 493)

    def test_missing_ink_does_not_shift_neighbor_values(self):
        image, boxes = survival_chart(omit_count=True)
        table = risk_count_table(image, boxes)
        self.assertEqual(table['status'], 'partial')
        self.assertEqual(table['rows'][2], ['Censored', '0','1','6', None, '13','16'])
        self.assertEqual([(cell['row'], cell['column']) for cell in table['review_cells']], [(3, 5)])

    def test_does_not_invent_missing_origin_tick(self):
        image, boxes = survival_chart(zero_tick=False)
        table = risk_count_table(image, boxes)
        self.assertTrue(table is None or '0' not in table['rows'][0])

    def test_time_axis_without_risk_rows_does_not_call_recognizer(self):
        image, boxes = survival_chart()
        boxes = [b for b in boxes if b[0] != 'Events']
        boxes.append(('Events', 250., 204., 340., 220., .99))
        with patch('slide_risk._recognize_cell') as read:
            self.assertIsNone(risk_count_table(image, boxes))
            read.assert_not_called()
