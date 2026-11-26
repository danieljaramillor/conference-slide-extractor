"""Build one text record per selected slide, with table evidence when available."""

import re
from bisect import bisect_right
from statistics import median

import cv2
import numpy as np


SESSION_TEXT = 'slides.jsonl'
_PAGE = re.compile(r'(?:slide|page)\s+(\d{1,4})(?:\s*(?:of|/)\s*\d{1,4})?', re.I)
_DATA = re.compile(r'[+−-]?\s*\d[\d.,%±/–−()\[\] -]*')


def _ocr_boxes(ocr):
    boxes = []
    for line in ocr.get('lines', []):
        text = line['text'].strip()
        if not text or line['score'] < .70:
            continue
        box = np.asarray(line['box'])
        left, top = box.min(axis=0)
        right, bottom = box.max(axis=0)
        if right > left and bottom > top:
            boxes.append((text, float(left), float(top), float(right), float(bottom), line['score']))
    return boxes


def _mid_y(box):
    return (box[2] + box[4]) / 2


def _printed_number(boxes, height):
    matches = {int(match[1]) for text, _, top, _, bottom, score in boxes
               if score >= .86 and (top >= .80 * height or bottom <= .16 * height)
               and (match := _PAGE.fullmatch(text.strip(' .')))}
    if len(matches) == 1:
        return matches.pop()
    return None


def _bands(projection, minimum):
    positions = np.flatnonzero(projection >= minimum)
    if not positions.size:
        return []
    return [int(values[len(values)//2]) for values in np.split(positions, np.flatnonzero(np.diff(positions) > 1)+1)]


def _native_bounds(xs, ys, scale, width, height):
    return [max(0, round(xs[0] / scale)), max(0, round(ys[0] / scale)),
            min(width, round((xs[-1]+1) / scale)), min(height, round((ys[-1]+1) / scale))]


def _ruled_tables(image, boxes):
    height, width = image.shape[:2]
    scale = min(1., 1440 / max(height, width))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if scale != 1.:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = gray.shape
    if min(h, w) < 100:
        return []
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 9)
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(26, w//27)), np.uint8))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(20, h//32), 1), np.uint8))
    # Anti-aliased intersections can have a small gap after downsampling.
    joint = cv2.dilate(cv2.bitwise_or(horizontal, vertical), np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(joint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tables = []
    for contour in contours:
        x, y, gw, gh = cv2.boundingRect(contour)
        if gw < w*.12 or gh < h*.065:
            continue
        xs = [x+i for i in _bands(np.count_nonzero(vertical[y:y+gh, x:x+gw], axis=0), .72*gh)]
        ys = [y+i for i in _bands(np.count_nonzero(horizontal[y:y+gh, x:x+gw], axis=1), .72*gw)]
        if not (3 <= len(xs) <= 31 and 3 <= len(ys) <= 61):
            continue
        if min(np.diff(xs)) < 12 or min(np.diff(ys)) < 9:
            continue
        populated, partial = {}, False
        for box in boxes:
            label, left, top, right, bottom, score = box
            cx, cy = (left+right)*scale/2, (top+bottom)*scale/2
            column, row = bisect_right(xs, cx)-1, bisect_right(ys, cy)-1
            if not (0 <= row < len(ys)-1 and 0 <= column < len(xs)-1):
                continue
            if left*scale < xs[column]-2 or right*scale > xs[column+1]+2 or score < .80:
                partial = True
                continue
            populated.setdefault((row,column), []).append(box)
        # A blank full-width division often separates two flowchart lanes.
        # Preserve those lanes in the raw text without treating the gap as a column.
        if len(populated) < max(4, (len(ys)-1)*(len(xs)-1)*.30) or len({col for _,col in populated}) < len(xs)-1:
            continue
        rows = []
        for row in range(len(ys)-1):
            values = []
            for col in range(len(xs)-1):
                cell = sorted(populated.get((row,col), []), key=lambda box: (round(_mid_y(box)/5), box[1]))
                values.append(' '.join(box[0] for box in cell) if cell else None)
            rows.append(values)
        tables.append({'bounds': _native_bounds(xs,ys,scale,width,height), 'method': 'ruled_grid',
                       'status': 'partial' if partial or any(None in row for row in rows) else 'reconstructed',
                       'rows': rows})
    return tables


def _aligned_tables(boxes, width):
    """Require repeated rows of separately recognized labels and numeric cells."""
    if len(boxes) < 9:
        return []
    typical = median(box[4]-box[2] for box in boxes)
    rows = []
    for box in sorted(boxes, key=_mid_y):
        if not rows or _mid_y(box)-_mid_y(rows[-1][0]) > .48*typical:
            rows.append([])
        rows[-1].append(box)
    tables, block = [], []

    def finish():
        if len(block) < 3:
            return
        left = min(cell[1] for row in block for cell in row)
        right = max(cell[3] for row in block for cell in row)
        if right-left < .13*width:
            return
        tables.append({'bounds': [round(left), round(min(cell[2] for cell in block[0])), round(right),
                                  round(max(cell[4] for cell in block[-1]))],
                       'method': 'aligned_ocr', 'status': 'inferred_rows',
                       'rows': [[cell[0] for cell in row] for row in block]})

    for row in rows:
        row.sort(key=lambda box: box[1])
        if (not 3 <= len(row) <= 15 or min(cell[5] for cell in row) < .86
                or not any(char.isalpha() for char in row[0][0])
                or any(not _DATA.fullmatch(cell[0]) for cell in row[1:])
                or any(a[3] > b[1] for a,b in zip(row,row[1:]))):
            finish()
            block = []
            continue
        if block:
            anchor = block[0]
            if (len(anchor) != len(row) or _mid_y(row[0])-_mid_y(block[-1][0]) > 2.6*typical
                    or abs(anchor[0][1]-row[0][1]) > typical
                    or any(min(abs(a[1]-b[1]), abs(a[3]-b[3]), abs(a[1]+a[3]-b[1]-b[3])/2) > .7*typical
                           for a,b in zip(anchor[1:],row[1:]))):
                finish()
                block = []
        block.append(row)
    finish()
    return tables


def slide_text_record(number, name, image, ocr):
    status = ocr.get('status', 'unavailable')
    boxes = _ocr_boxes(ocr) if status == 'ok' else []
    tables = _ruled_tables(image, boxes) if len(boxes) >= 4 else []
    # Avoid inferring a second copy of rows already inside a detected grid.
    unassigned = [box for box in boxes if not any(
        left <= (box[1]+box[3])/2 <= right and top <= _mid_y(box) <= bottom
        for left,top,right,bottom in (table['bounds'] for table in tables))]
    tables.extend(_aligned_tables(unassigned, image.shape[1]))
    return {'slide_number': number, 'printed_slide_number': _printed_number(boxes, image.shape[0]),
            'source': name, 'ocr_status': status, 'text': ocr['text'] if status == 'ok' else None,
            'tables': sorted(tables, key=lambda table: (table['bounds'][1], table['bounds'][0]))}
