"""Build one text record per selected slide, with table evidence when available."""

import re
from bisect import bisect_left, bisect_right
from statistics import median

import cv2
import numpy as np


SESSION_TEXT = 'slides.jsonl'
_PAGE = re.compile(r'(?:slide|page)\s+(\d{1,4})(?:\s*(?:of|/)\s*\d{1,4})?', re.I)
_DATA = re.compile(r'[+−-]?\s*\d[\d.,%±/–−()\[\] -]*')
_SINGLE_NUMBER = re.compile(r'[+−-]?\s*\d+(?:[.,]\d+)*\s*%?')
_REREAD_LIMIT = 8


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
    return [(int(values[0]), int(values[-1])+1)
            for values in np.split(positions, np.flatnonzero(np.diff(positions) > 1)+1)]


def _native_bounds(bounds, scales):
    return [round(position / scale) for position, scale in zip(bounds, scales * 2)]


def _cell_ink_crop(image, text_ink, bounds, scales):
    """Crop an isolated line of ink with room on every side inside its cell."""
    left, top, right, bottom = bounds
    mask = text_ink[top:bottom, left:right]
    x, y, width, height = cv2.boundingRect(mask)
    if (width < 3 or height < 4 or min(x, y, mask.shape[1]-x-width, mask.shape[0]-y-height) < 4
            or not mask[y:y+height].any(axis=1).all()):
        return None
    x1, y1, x2, y2 = _native_bounds((left+x-2, top+y-2, left+x+width+2, top+y+height+2), scales)
    return image[y1:y2, x1:x2]


def _reread_number(crop):
    try:
        from slide_ocr import _recognize_cell

        label, confidence = _recognize_cell(crop)
    except (ImportError, OSError, RuntimeError, ValueError, cv2.error):
        return None
    label = label.strip()
    if confidence >= .95 and _SINGLE_NUMBER.fullmatch(label):
        return label, round(confidence, 4)
    return None


def _grid_rows(image, boxes, ink, x_bands, y_bands, scales, remaining):
    sx, sy = scales
    xs = [(start+end)//2 for start, end in x_bands]
    ys = [(start+end)//2 for start, end in y_bands]
    populated, uncertain, supported = {}, {}, set()
    for label, left, top, right, bottom, score in boxes:
        left, top, right, bottom = left*sx, top*sy, right*sx, bottom*sy
        column = bisect_right(xs, (left+right)/2)-1
        row = bisect_right(ys, (top+bottom)/2)-1
        if 0 <= row < len(ys)-1 and 0 <= column < len(xs)-1:
            supported.add((row, column))
            contained = left >= xs[column] and right <= xs[column+1] and top >= ys[row] and bottom <= ys[row+1]
            if contained and score >= .80:
                populated.setdefault((row, column), []).append((label, top+bottom, left))
                continue
        columns = range(max(0, bisect_right(xs, left)-1), min(len(xs)-1, bisect_left(xs, right)))
        rows = range(max(0, bisect_right(ys, top)-1), min(len(ys)-1, bisect_left(ys, bottom)))
        for r in rows:
            for c in columns:
                contained = left >= xs[c] and right <= xs[c+1] and top >= ys[r] and bottom <= ys[r+1]
                reasons = uncertain.setdefault((r, c), set())
                reasons.add('low_ocr_confidence' if contained else 'crosses_cell_boundary')

    # A blank division often separates two flowchart lanes. Its center has no
    # textual support, even when the detected rules resemble a table.
    if len(supported) < max(4, (len(ys)-1)*(len(xs)-1)*.30) or len({c for _,c in supported}) < len(xs)-1:
        return None, remaining

    rows = [[None]*(len(xs)-1) for _ in range(len(ys)-1)]
    for (row, column), reads in populated.items():
        ordered = sorted(reads, key=lambda read: (round(read[1]/10), read[2]))
        rows[row][column] = ' '.join(read[0] for read in ordered)

    review_cells, cell_sources = [], []
    text_ink = None
    if uncertain and remaining:
        # Remove only the confirmed ruling strips. The line detector also
        # finds long strokes in cell content, which must stay in the ink mask.
        text_ink = ink.copy()
        for start, end in x_bands:
            text_ink[:, start:end] = 0
        for start, end in y_bands:
            text_ink[start:end, :] = 0
    for (row, column), reasons in sorted(uncertain.items()):
        bounds = (xs[column], ys[row], xs[column+1], ys[row+1])
        location = {'row': row+1, 'column': column+1, 'bounds': _native_bounds(bounds, scales)}
        previous = rows[row][column]
        rows[row][column] = None
        recovered = None
        if remaining and (previous is None or _SINGLE_NUMBER.fullmatch(previous)):
            crop = _cell_ink_crop(image, text_ink, bounds, scales)
            if crop is not None:
                recovered = _reread_number(crop)
                remaining -= 1
        if recovered and (previous is None or previous.replace(' ', '') == recovered[0].replace(' ', '')):
            rows[row][column] = recovered[0]
            cell_sources.append({**location, 'method': 'native_reread', 'confidence': recovered[1]})
        else:
            reason = 'crosses_cell_boundary' if 'crosses_cell_boundary' in reasons else 'low_ocr_confidence'
            review_cells.append({**location, 'reason': reason})
    return (rows, review_cells, cell_sources), remaining


def _ruled_tables(image, boxes):
    height, width = image.shape[:2]
    scale = min(1., 1440 / max(height, width))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if scale != 1.:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = gray.shape
    if min(h, w) < 100:
        return [], []
    scales = (w/width, h/height)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 9)
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(26, w//27)), np.uint8))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(20, h//32), 1), np.uint8))
    # Anti-aliased intersections can have a small gap after downsampling.
    joint = cv2.dilate(cv2.bitwise_or(horizontal, vertical), np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(joint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tables, grid_bounds = [], []
    remaining = _REREAD_LIMIT
    for contour in contours:
        x, y, gw, gh = cv2.boundingRect(contour)
        if gw < w*.12 or gh < h*.065:
            continue
        x_bands = [(x+start, x+end) for start, end in
                   _bands(np.count_nonzero(vertical[y:y+gh, x:x+gw], axis=0), .72*gh)]
        y_bands = [(y+start, y+end) for start, end in
                   _bands(np.count_nonzero(horizontal[y:y+gh, x:x+gw], axis=1), .72*gw)]
        if not (3 <= len(x_bands) <= 31 and 3 <= len(y_bands) <= 61):
            continue
        xs = [(start+end)//2 for start, end in x_bands]
        ys = [(start+end)//2 for start, end in y_bands]
        if min(np.diff(xs)) < 12 or min(np.diff(ys)) < 9:
            continue
        bounds = _native_bounds((xs[0], ys[0], xs[-1]+1, ys[-1]+1), scales)
        grid_bounds.append(bounds)
        recovered, remaining = _grid_rows(image, boxes, ink, x_bands, y_bands, scales, remaining)
        if recovered is not None:
            rows, review_cells, cell_sources = recovered
            tables.append({'bounds': bounds, 'method': 'ruled_grid',
                           'status': 'partial' if any(None in row for row in rows) else 'reconstructed',
                           'rows': rows, 'review_cells': review_cells, 'cell_sources': cell_sources})
    return tables, grid_bounds


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
    tables, grid_bounds = _ruled_tables(image, boxes) if len(boxes) >= 4 else ([], [])
    # Inferred rows must not reuse an OCR box rejected by the physical grid.
    unassigned = [box for box in boxes if not any(
        left < box[3] and box[1] < right and top < box[4] and box[2] < bottom
        for left,top,right,bottom in grid_bounds)]
    tables.extend(_aligned_tables(unassigned, image.shape[1]))
    return {'slide_number': number, 'printed_slide_number': _printed_number(boxes, image.shape[0]),
            'source': name, 'ocr_status': status, 'text': ocr['text'] if status == 'ok' else None,
            'tables': sorted(tables, key=lambda table: (table['bounds'][1], table['bounds'][0]))}
