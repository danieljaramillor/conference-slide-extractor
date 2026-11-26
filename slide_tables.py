"""Recover rows from independently aligned labels and values within a table."""

import re
from statistics import median


_VALUE = re.compile(r'(?:[<>≤≥]?\s*|[nN]\s*=\s*)[+−-]?(?:\d|\.\d)[\d\s.,:%±/–−()\[\];+*†≥≤<>-]*')
_KEYED_VALUE = re.compile(r'[\w -]{1,18}:\s*\d.*[%()]')
_WRAPPED_VALUE = re.compile(_KEYED_VALUE.pattern+r'\s*[;,.]?')
_UNITS = re.compile(r'(?:n\s*(?:\(.+\))?|%|(?:months?|years?)\s*[\[(].+[\])])', re.I)
_PLAIN_NUMBER = re.compile(r'[+−-]?\d+(?:\.\d+)?')
_COUNT_HEADER = re.compile(r'n\s*=\s*\d+', re.I)


def value_text(text):
    return bool(_VALUE.fullmatch(text) or (len(text) <= 68 and _KEYED_VALUE.fullmatch(text)))


def _label(box):
    text = box[0]
    return (not value_text(text) and not _UNITS.fullmatch(text) and not text.startswith(('•', '·', '▪'))
            and any(char.isalpha() for char in text))


def _x(box):
    return (box[1] + box[3]) / 2


def _y(box):
    return (box[2] + box[4]) / 2


def _tracks(values, pitch):
    """Find stable left, center, or right anchoring, including padded OCR boxes."""
    candidates = set()
    for position in (_x, lambda box: box[1], lambda box: box[3]):
        ordered = sorted(values, key=position)
        for index, first in enumerate(ordered):
            track = []
            for box in ordered[index:]:
                if position(box) - position(first) > pitch:
                    break
                track.append(box)
            if len(track) >= 3:
                candidates.add(tuple(sorted(track, key=_y)))
    return sorted(candidates, key=lambda track: (-len(track), median(_x(box) for box in track), track))


def _axis(track):
    if len(track) < 6 or not all(_PLAIN_NUMBER.fullmatch(box[0]) for box in track):
        return False
    numbers = [float(box[0].replace('−', '-')) for box in track]
    steps = [b - a for a, b in zip(numbers, numbers[1:])]
    step = median(steps)
    return bool(step and sum(abs(s - step) < abs(step)*.01 for s in steps) >= (len(steps)-1)*.70)


def _runs(track, typical):
    groups = []
    for box in track:
        if not groups or _y(box) - _y(groups[-1][-1]) > 3.65*typical:
            groups.append([])
        groups[-1].append(box)
    return [group for group in groups if len(group) >= 3]


def _same_row(a, b, typical):
    return (abs(_y(a) - _y(b)) <= typical*.40
            and min(a[4], b[4]) - max(a[2], b[2]) >= min(a[4]-a[2], b[4]-b[2])*.40)


def _label_matches(track, labels, typical):
    matches = []
    for value in track:
        # Labels closest to the first value lane help avoid bullets in a
        # neighboring panel. Use a separate label track to support the region.
        nearby = [label for label in labels if label[3] < value[1] - typical*.16 and _same_row(value, label, typical)]
        matches.append(max(nearby, key=lambda label: label[3], default=None))
    return matches


def _near_column(a, b, typical):
    return abs(_x(a) - _x(b)) <= typical*1.50 or abs(a[3] - b[3]) <= typical*.80 or abs(a[1]-b[1]) <= typical*.80


def _expanded_tracks(track, available, typical):
    lanes = [track]
    for candidate in available:
        if _axis(candidate) or any(_near_column(candidate[len(candidate)//2], lane[len(lane)//2], typical)
                                  for lane in lanes):
            continue
        consistent = [box for box in candidate if any(_same_row(box, main, typical) for main in track)]
        if len(consistent) < max(3, len(track)*.60):
            continue
        if any(a[1] < b[3] and b[1] < a[3] for a in consistent for lane in lanes for b in lane
               if _same_row(a,b,typical)):
            continue
        lanes.append(consistent)
    return sorted(lanes, key=lambda lane: median(_x(box) for box in lane))


def _union_bounds(boxes):
    if not boxes:
        return None
    return [round(min(b[1] for b in boxes)), round(min(b[2] for b in boxes)),
            round(max(b[3] for b in boxes)), round(max(b[4] for b in boxes))]


def _wrapped_value(head, boxes, label, edges, typical, data_rows):
    if not label or not _KEYED_VALUE.fullmatch(head[0]):
        return [head]
    left, right = edges
    neighbors = [box for box in boxes if box != head and box not in data_rows and _WRAPPED_VALUE.fullmatch(box[0])
                 and left <= _x(box) <= right and abs(_x(head)-_x(box)) <= typical*.60
                 and abs(_y(box)-_y(label)) <= typical*.55 and abs(_y(head)-_y(label)) <= typical*.55
                 and abs(_y(box)-_y(head)) >= typical*.55]
    if len(neighbors) == 1:
        return sorted([head, neighbors[0]], key=_y)
    return [head]


def _make_table(track, available, boxes, labels, typical, width, grids):
    # A lane just beyond the right edge of a detected grid can complete its
    # missing outer column. The grid also excludes neighboring content.
    center = median(_x(b) for b in track)
    hints = [g for g in grids if g[2] < center < g[2]+.16*width
             and sum(g[1]-typical*.40 <= _y(b) <= g[3]+typical*.40 for b in track) >= len(track)*.8]
    if len(hints) == 1:
        x1,y1,x2,y2 = hints[0]
        boxes = [b for b in boxes if _x(b) >= x1-typical*.25 and y1-typical*.65 <= _y(b) <= y2+typical*.45]
        labels = [b for b in labels if b in boxes]
        track = [b for b in track if b in boxes]
    available = [[b for b in lane if b in boxes] for lane in available]
    available = [lane for lane in available if len(lane) >= 3]
    if len(track) < 3:
        return None
    match = _label_matches(track, labels, typical)
    if match[0] is None and _COUNT_HEADER.fullmatch(track[0][0]):
        track, match = track[1:],match[1:]
    paired = [box for box in match if box]
    if len(paired) < max(3, len(track)*.65):
        return None
    # A stable left label edge is a required anchor. Allow sublabels and
    # indented rows elsewhere inside the resulting rectangle.
    if max(sum(abs(a[1]-b[1]) <= typical*1.25 for a in paired) for b in paired) < max(3, len(paired)*.45):
        return None
    left = min(box[1] for box in paired)
    if len(hints)==1:
        left = min(left,float(hints[0][0]))
    lanes = _expanded_tracks(track, available, typical)
    left_lanes = [lane for lane in lanes if median(_x(b) for b in lane) > left + typical*2]
    if not left_lanes or track not in left_lanes:
        return None
    right = max(box[3] for lane in left_lanes for box in lane)
    if right-left < .13*width:
        return None
    # Midpoints separate recognized column footprints. A box that spans two
    # actual numeric lanes stays in the slide text instead of supplying a cell.
    centers = [median(_x(box) for box in lane) for lane in left_lanes]
    x_edges = [max(box[3] for box in paired) + typical*.14]
    first_left = median(box[1] for box in left_lanes[0])
    x_edges[0] = min(x_edges[0], first_left - typical*.15)
    x_edges.extend((a+b)/2 for a,b in zip(centers,centers[1:]))
    x_edges.append(right + typical*.26)
    if x_edges[0] <= left or any(a >= b for a,b in zip(x_edges,x_edges[1:])):
        return None

    selected = [box for box in boxes if left-typical*.65 <= _x(box) <= x_edges[-1]
                and _y(track[0])-typical*3.2 <= _y(box) <= _y(track[-1])+typical*.42
                and box[3] - box[1] < right-left + typical*.4 and box[4]-box[2] <= typical*1.65]
    rows, row_sources, bounds_by_row, units = [], [], [], []
    claimed = set()
    data_reads = {box for lane in left_lanes for box in lane}
    for source_row, main in enumerate(track):
        if source_row and _y(main) - _y(track[source_row-1]) < typical*.57:
            return None
        row = [box for box in selected if _same_row(main,box,typical) and box not in claimed]
        claimed.update(row)
        label_boxes = sorted((box for box in row if _x(box) < x_edges[0] and not _UNITS.fullmatch(box[0])),
                             key=lambda box: box[1])
        fields = [' '.join(b[0] for b in label_boxes) or None]
        bounds = [_union_bounds(label_boxes)]
        unit_boxes = [box for box in row if _UNITS.fullmatch(box[0]) and
                      (not label_boxes or box[1] > max(b[3] for b in label_boxes)) and
                      box[3] < median(b[1] for b in left_lanes[0])]
        for c, lane in enumerate(left_lanes):
            options = [box for box in lane if box in row and x_edges[c] <= _x(box) < x_edges[c+1]]
            # A midpoint cannot authorize a numeric read across columns.
            options = [b for b in options if b[1] >= x_edges[c] - typical*.30 and b[3] <= x_edges[c+1] + typical*.30]
            numeric = (_wrapped_value(options[0], selected, label_boxes[-1] if label_boxes else None,
                                      x_edges[c:c+2], typical, data_reads) if len(options)==1 else [])
            fields.append(' '.join(b[0] for b in numeric) or None)
            bounds.append(_union_bounds(numeric))
        if fields[0] and any(fields[1:]):
            rows.append(fields)
            row_sources.append(main)
            bounds_by_row.append(bounds)
            units.append(unit_boxes[0] if len(unit_boxes)==1 else None)
    if len(rows) < max(3, len(track)*.65):
        return None

    # Include aligned text headers only within the detected body rectangle and
    # in the immediately preceding band. A spanning header stays in raw OCR.
    prefix = sorted((b for b in selected if b not in claimed and (not value_text(b[0]) or _COUNT_HEADER.fullmatch(b[0]))
                     and _y(b) <= _y(row_sources[0])-typical*.60 and not b[0].startswith(('•','·'))),
                    key=lambda b: (_y(b), b[1]))
    header = [None]*len(rows[0])
    header_sources = []
    header_bounds = [None]*len(header)
    if prefix:
        nearest = next((b for b in reversed(prefix) if _x(b) >= x_edges[0]),prefix[-1])
        for box in prefix:
            if not _same_row(nearest,box,typical):
                continue
            index = sum(_x(box) >= x for x in x_edges[:-1])
            if not header[index]:
                header[index] = box[0]
                header_sources.append(box)
                header_bounds[index] = _union_bounds([box])
    unit_anchors = [box for box in units if box]
    column_bounds = [round(left), *map(round, x_edges[:-1]), round(right)]
    has_units = len(unit_anchors) >= 3 and max(_x(b) for b in unit_anchors)-min(_x(b) for b in unit_anchors) <= typical*1.10
    if has_units:
        for row, bounds, unit in zip(rows, bounds_by_row, units):
            row.insert(1,unit[0] if unit else None)
            bounds.insert(1,_union_bounds([unit]) if unit else None)
        header.insert(1,None)
        header_bounds.insert(1,None)
        unit_left = min(x_edges[0],median(b[1] for b in unit_anchors))
        column_bounds[1:2] = [round(unit_left), round(max(x_edges[0],max(b[3] for b in unit_anchors)))]
    if sum(bool(name) for name in header) >= 2 or any(header[1:]):
        rows.insert(0,header)
        bounds_by_row.insert(0,header_bounds)
    else:
        header_sources = []
    groups = []
    preceding = [b for b in prefix if header_sources and
                 .60*typical <= min(_y(h) for h in header_sources)-_y(b) <= 2.0*typical
                 and not _UNITS.fullmatch(b[0])]
    for box in preceding:
        columns = [c+2+int(has_units) for c, center in enumerate(centers)
                   if box[1]-typical*.45 <= center <= box[3]+typical*.45]
        if len(columns) >= (2 if len(centers)>1 else 1) and any(char.isalpha() for char in box[0]):
            groups.append({'text':box[0], 'first_column':columns[0], 'last_column':columns[-1],
                           'bounds':_union_bounds([box])})
    top = min([row_sources[0][2], *(box[2] for box in header_sources), *(g['bounds'][1] for g in groups)])
    bottom = max(box[4] for box in row_sources)
    table = {'bounds': [max(0, round(left)), round(top), round(right), round(bottom)],
             'method':'aligned_ocr', 'status':'inferred_rows', 'rows':rows,
             'column_bounds':column_bounds, 'cell_bounds':bounds_by_row}
    if groups:
        table['column_groups'] = groups
    return table


def _overlap(a,b):
    left,top,right,bottom = a
    intersection = max(0, min(right,b[2])-max(left,b[0]))*max(0,min(bottom,b[3])-max(top,b[1]))
    return intersection / max(1,min((right-left)*(bottom-top),(b[2]-b[0])*(b[3]-b[1])))


def aligned_tables(boxes, width, grids=(), ruled=()):
    """Find data rectangles before reading the cells, so nearby prose cannot shift rows."""
    if len(boxes) < 6:
        return []
    typical = median(b[4]-b[2] for b in boxes)
    values = [b for b in boxes if b[5]>=.86 and typical*.45 <= b[4]-b[2] <= typical*1.65 and value_text(b[0])]
    labels = [b for b in boxes if b[5]>=.80 and b[4]-b[2] <= typical*1.65 and _label(b)]
    runs = {tuple(run) for track in _tracks(values,typical*.85) for run in _runs(track,typical)}
    axes = {box for run in runs if _axis(run) for box in run}
    # A declined drawn grid may be a chart. Its tick labels cannot become
    # columns in a neighboring OCR-aligned table.
    declined = [g for g in grids if not any(g == t['bounds'] for t in ruled)]
    axes.update(b for b in values if _PLAIN_NUMBER.fullmatch(b[0]) and
                any(g[0] <= _x(b) <= g[2] and g[1] <= _y(b) <= g[3] for g in declined))
    remaining = {tuple(b for b in run if b not in axes) for run in runs}
    tracks = [list(run) for run in sorted(remaining,key=lambda r:(-len(r),r)) if len(run)>=3]
    candidates = [table for track in tracks if (table:=_make_table(track,tracks,boxes,labels,typical,width,grids))]
    candidates.sort(key=lambda t: -sum(v is not None for row in t['rows'] for v in row[1:]))
    result = []
    for table in candidates:
        if not any(_overlap(table['bounds'],other['bounds']) >= .45 for other in result):
            result.append(table)
    return result


def combine_tables(ruled, grid_bounds, aligned):
    """Prefer a physical grid unless it omitted a supported column or most reads."""
    tables = list(ruled)
    for table in aligned:
        bounds = table['bounds']
        grids = [g for g in grid_bounds if _overlap(bounds,g) >= .35]
        if len(grids) > 1:
            continue
        if grids:
            grid, = grids
            physical = next((candidate for candidate in ruled if candidate['bounds']==grid),None)
            extends_right = (bounds[2]-grid[2] >= (grid[2]-grid[0])*.10 and bounds[0] >= grid[0]-20)
            sparse = (physical is not None and not physical['cell_sources'] and
                      sum(value is not None for row in physical['rows'] for value in row) < len(physical['rows']) and
                      _overlap(bounds,grid) >= .80)
            if not (extends_right or sparse):
                continue
            if physical in tables:
                tables.remove(physical)
        tables.append(table)
    return tables
