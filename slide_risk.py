"""Read compact at-risk, censor, and event rows under a survival chart."""

import re

import cv2
import numpy as np

from slide_ocr import _recognize_cell


_TIME_CAPTION = re.compile(
    r'(?:(?:follow[\s-]*up(?:\s+time)?|time)(?:\s+since\s+[^()]{1,28})?\s*)?'
    r'\(?(?:days?|weeks?|months?|years?)\)?', re.I
)
_INTEGER = re.compile(r'(?:0|[1-9][0-9]{0,5})')
_ROLES = {
    'atrisk': 'At risk', 'numberatrisk': 'At risk', 'noatrisk': 'At risk',
    'patientsatrisk': 'At risk', 'censored': 'Censored', 'numbercensored': 'Censored',
    'events': 'Events', 'numberofevents': 'Events', 'noofevents': 'Events',
}


def _role(label):
    return _ROLES.get(re.sub(r'[\s.:#-]+', '', label.casefold()))


def _x(box):
    return (box[1] + box[3]) / 2


def _y(box):
    return (box[2] + box[4]) / 2


def _axis_ticks(boxes, caption, width):
    height = caption[4] - caption[2]
    nearby = [b for b in boxes if b[5] >= .90 and _INTEGER.fullmatch(b[0])
              and caption[2] - height * 1.1 <= _y(b) <= caption[2] + height * .35]
    if not 4 <= len(nearby) <= 18:
        return None
    ticks = sorted(nearby, key=_x)
    xs = [_x(b) for b in ticks]
    values = [int(b[0]) for b in ticks]
    places = np.diff(xs)
    steps = np.diff(values)
    pitch = float(np.median(places))
    middle = float(np.median([_y(b) for b in ticks]))
    if (xs[-1] - xs[0] < width * .29 or not 16 <= pitch <= width * .30
            or not xs[0] < _x(caption) < xs[-1]
            or steps[0] <= 0 or not np.all(steps == steps[0])
            or not np.all(np.abs(places - pitch) <= pitch * .14)
            or any(abs(_y(b) - middle) > height * .30 for b in ticks)):
        return None
    return ticks, xs, pitch, int(steps[0]), middle


def _ink(gray):
    if gray.size < 4 or np.ptp(gray) < 24:
        return np.zeros(gray.shape, np.uint8)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


def _read_patch(image, gray, bounds, limit=.93):
    """Tighten around the isolated glyphs before recognizer resize and padding."""
    left, top, right, bottom = bounds
    mask = _ink(gray[top:bottom, left:right])
    x, y, w, h = cv2.boundingRect(mask)
    if h < 4 or w < 2 or np.count_nonzero(mask) > mask.size * .56:
        return '', 0.
    x1, y1 = max(left, left+x-1), max(top, top+y-1)
    x2, y2 = min(right, left+x+w+1), min(bottom, top+y+h+1)
    patch = cv2.copyMakeBorder(image[y1:y2, x1:x2], 2, 2, 4, 4, cv2.BORDER_REPLICATE)
    try:
        text, confidence = _recognize_cell(patch)
        if confidence >= limit:
            return text.strip(), confidence
    except (ImportError, OSError, RuntimeError, ValueError, cv2.error):
        pass
    return '', 0.


def _column_bounds(x, pitch, width):
    return max(0, round(x-pitch*.35)), min(width, round(x+pitch*.30))


def _row_bands(gray, columns, top, bottom, tick_height):
    mask = _ink(gray[top:bottom, columns[0][0]:columns[-1][1]])
    if not mask.size or np.count_nonzero(mask) > .30 * mask.size:
        return []
    offset = columns[0][0]
    support = sum(mask[:, left-offset:right-offset].any(axis=1).astype(int) for left, right in columns)
    occupied = np.flatnonzero(support >= max(3, len(columns)//2)) + top
    return [(int(part[0]), int(part[-1]+1))
            for part in np.split(occupied, np.flatnonzero(np.diff(occupied) > 1)+1)
            if max(3, tick_height*.18) <= len(part) <= tick_height*.86]


def _read_strip(image, caption, role_hints, axis):
    height, width = image.shape[:2]
    ticks, ticks_x, pitch, step, middle = axis
    tick_height = float(np.median([b[4]-b[2] for b in ticks]))
    top = round(caption[4]+1)
    bottom = min(height, round(top + tick_height*4.4))
    if not any(top <= _y(b) < bottom and b[3] < ticks_x[0]-pitch*.30 for b in role_hints):
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    header = [b[0] for b in ticks]
    header_bounds = [[round(v) for v in b[1:5]] for b in ticks]
    if int(header[0]) == step and ticks_x[0]-pitch > pitch*.50:
        x = ticks_x[0]-pitch
        left, right = _column_bounds(x, pitch*.48, width)
        guess = [left, max(0, round(middle-tick_height*.34)), right, min(height, round(middle+tick_height*.34))]
        text, _ = _read_patch(image, gray, guess, limit=.94)
        if text == '0':
            ticks_x = [x, *ticks_x]
            header.insert(0, text)
            header_bounds.insert(0, guess)

    columns = [_column_bounds(x, pitch, width) for x in ticks_x]
    label_left = max(0, round(ticks_x[0]-pitch*1.9))
    label_right = round(ticks_x[0]-pitch*.36)
    bands = _row_bands(gray, columns, top, bottom, tick_height)
    if not 2 <= len(bands) <= 9 or label_right-label_left < 8:
        return None

    labeled = []
    roles = set()
    for first, last in bands:
        if labeled and first - labeled[-1][1] > tick_height * .70:
            break
        # Keep the recognizer crop inside its own ink band plus a single pixel
        # on either side. Even a small margin can include a neighboring row.
        label_box = [label_left, max(top, first-1), label_right, min(bottom, last+1)]
        text, _ = _read_patch(image, gray, label_box, limit=.87)
        role = _role(text)
        if role is None:
            if labeled:
                break
            continue
        if role in roles or (not labeled and role != 'At risk'):
            return None
        roles.add(role)
        labeled.append((role, last, label_box))
        if len(labeled) == 3:
            break
    if len(labeled) < 2:
        return None

    rows = [[caption[0], *header]]
    coords = [[[round(v) for v in caption[1:5]], *header_bounds]]
    sources, review = [], []
    for row_number, (role, _, label_box) in enumerate(labeled, 2):
        row = [role]
        row_boxes = [label_box]
        for column, (left, right) in enumerate(columns, 2):
            crop = [left, label_box[1], right, label_box[3]]
            text, confidence = _read_patch(image, gray, crop)
            cell = {'row': row_number, 'column': column, 'bounds': crop}
            if _INTEGER.fullmatch(text):
                row.append(text)
                sources.append({**cell, 'method': 'native_reread', 'confidence': round(confidence, 4)})
            else:
                row.append(None)
                review.append({**cell, 'reason': 'unreadable_risk_count'})
            row_boxes.append(crop)
        rows.append(row)
        coords.append(row_boxes)

    if any(sum(v is not None for v in row[1:]) < len(columns)*.70 for row in rows[1:]):
        return None
    return {'bounds': [label_left, min(b[1] for b in header_bounds), columns[-1][1], coords[-1][0][3]],
            'method': 'risk_count_strip', 'status': 'partial' if review else 'reconstructed',
            'rows': rows, 'cell_bounds': coords, 'cell_sources': sources, 'review_cells': review}


def risk_count_table(image, boxes):
    """Return a time-aligned risk strip only with independently read row labels."""
    roles = [b for b in boxes if b[5] >= .80 and _role(b[0])]
    if not roles:
        return None
    width = image.shape[1]
    captions = [b for b in boxes if b[5] >= .87 and b[3]-b[1] < width*.38 and _TIME_CAPTION.fullmatch(b[0])]
    if len(captions) != 1:
        return None
    caption, = captions
    axis = _axis_ticks(boxes, caption, width)
    return _read_strip(image, caption, roles, axis) if axis else None
