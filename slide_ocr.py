"""Offline OCR, cache, and conservative text evidence for slide matching."""
from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import threading
import unicodedata
import uuid

import numpy as np
import cv2

ENGINE_VERSION='rapidocr-3.10-v6-small-det960-v1'
HIGH_RES_ENGINE_VERSION='rapidocr-3.10-v6-small-det1536-v1'
_COMPACT_CITATION = re.compile(r',\d+\.[A-Za-z]')
_CITATION_SEPARATOR = re.compile(r',\s*(?=\d{1,2}\.\s*[A-Za-z])')
_MONTH_DATE = re.compile(r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|'
                         r'Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?) '
                         r'\d{1,2},\s*(?:19|20)\d{2}\b', re.I)
_local=threading.local()
_NUMBER = re.compile(r'\d+(?:[.,]\d+)?')
_SIGNED_NUMBER = re.compile(r'(?<![\w.,])[-+−]\s*\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?')
_COUNT_THEN_PERCENT = re.compile(r'(\d+\s*/\s*\d+)\s*,\s*(?=\d+(?:[.,]\d+)?\s*%)')
_BIBLIOGRAPHY_PAGES = re.compile(r'(\b(?:19|20)\d{2}\s*[:;, ]\s*\d+\s*:\s*\d+)\s*–\s*-\s*(?=\d+\b)')
_CITATION = re.compile(r'(?<=\w)[\{\^]\d{1,4}\s*$')
_FUSED_CITATION = re.compile(r'\b([A-Za-z][A-Za-z0-9]{2,}/\d)([ilI])?(\d{1,2})\s*$')
_REFERENCE_TEXT = re.compile(r'\b(?:publications?|references?|citations?|bibliography)\b', re.I)
_REFERENCE_LIST = re.compile(r'\d{1,2},\s*\d{1,2}\s*$')
_AFFILIATION = re.compile(r'(?<![^\W\d_])(?:hospital|universit[^\W\d_]*|institute|department|college|cent(?:er|re)|oncology|ltd)\b', re.I)
_AFFILIATION_MARKER = re.compile(r'(^|;\s*)(?:\d{1,2}�?|[¹²³⁴⁵⁶⁷⁸⁹⁰]{1,2}|&)(?=[^\W\d_])')


def engine(*, high_res=False):
    attribute = 'high_res_ocr' if high_res else 'ocr'
    if not hasattr(_local, attribute):
        import rapidocr
        from rapidocr import RapidOCR
        models=Path(rapidocr.__file__).parent/'models'
        paths={'Det.model_path':models/'PP-OCRv6_det_small.onnx',
               'Rec.model_path':models/'PP-OCRv6_rec_small.onnx',
               'Cls.model_path':models/'ch_ppocr_mobile_v2.0_cls_mobile.onnx'}
        for path in paths.values():
            if not path.is_file():
                raise RuntimeError(f'Missing bundled OCR model {path}. Reinstall rapidocr==3.10.0.')
        reader = RapidOCR(params={**{k:str(v) for k,v in paths.items()},
            'EngineConfig.onnxruntime.intra_op_num_threads':1,
            'EngineConfig.onnxruntime.inter_op_num_threads':1,
            'Det.limit_side_len':1536 if high_res else 960,'Det.limit_type':'max',
            'Global.max_side_len':4096,'Global.log_level':'error'})
        setattr(_local, attribute, reader)
    return getattr(_local, attribute)


def read_text(image, cache_dir):
    return _read_text(image, cache_dir, ENGINE_VERSION, engine)


def _read_text(image, cache_dir, version, get_engine):
    key=hashlib.sha256(version.encode()+str(image.shape).encode()+image.tobytes()).hexdigest()
    cache_dir=Path(cache_dir)
    cache_dir.mkdir(parents=True,exist_ok=True)
    path=cache_dir/f'{key}.json'
    if path.exists():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (OSError,ValueError):
            pass
    result=get_engine()(image,use_det=True,use_cls=False,use_rec=True,text_score=.50)
    lines=[]
    if result.txts is not None:
        for text,score,box in zip(result.txts,result.scores,result.boxes):
            pts=np.asarray(box,float)
            lines.append({'text':str(text),'score':round(float(score),4),'box':pts.tolist(),
                          'height':round(float((np.linalg.norm(pts[3]-pts[0])+np.linalg.norm(pts[2]-pts[1]))/2),2)})
    output=_text_result(lines, version)
    # Each thread writes a different temporary filename, then atomically replaces.
    temp=path.with_suffix(f'.{uuid.uuid4().hex}.tmp')
    temp.write_text(json.dumps(output,ensure_ascii=False),encoding='utf-8')
    temp.replace(path)
    return output


def _text_result(lines, version):
    strong = [line for line in lines if line['score'] >= .75]
    return {'status': 'ok', 'engine': version, 'lines': lines,
            'text': '\n'.join(line['text'] for line in lines),
            'characters': sum(len(line['text']) for line in strong),
            'mean_confidence': round(float(np.mean([line['score'] for line in strong])), 3) if strong else 0.,
            'median_text_height': round(float(np.median([line['height'] for line in strong])), 2) if strong else 0.}


def _export_text_needs_higher_resolution(image, lines):
    long_edge = max(image.shape[:2])
    if long_edge <= 960:
        return False
    # At the standard detector size, several lines this small indicate that
    # nearby labels or counts may have disappeared during detection.
    small = [line for line in lines if 0 < line.get('height', 0) * 960 / long_edge <= 17]
    return len(small) >= 8 or any(_COMPACT_CITATION.search(line['text']) for line in small)


def _text_bounds(lines):
    points = np.asarray([line['box'] for line in lines])
    return points.min(axis=1), points.max(axis=1)


def _numbers_for_spacing(text):
    text = _MONTH_DATE.sub(lambda match: match[0].replace(',', ' '), text)
    if re.match(r'^\d{1,2}\.\s*\w', text):
        # In a bibliography a comma can mark the start of the next numbered
        # reference. Treat it as a separator in both readings.
        text = _CITATION_SEPARATOR.sub(', ', text)
    return _values(numeric_text(text))


def _restores_word_boundaries(original, detailed):
    return (''.join(original.split()) == ''.join(detailed.split())
            and len(detailed.split()) > len(original.split())
            and _numbers_for_spacing(original) == _numbers_for_spacing(detailed))


def _merge_export_lines(original, detailed):
    if not original or not detailed:
        return original, 0, 0
    first, last = _text_bounds(original)
    detail_first, detail_last = _text_bounds(detailed)
    start = np.maximum(first[:, None], detail_first)
    end = np.minimum(last[:, None], detail_last)
    common = np.maximum(end - start, 0).prod(axis=2)
    smaller = np.minimum((last - first).prod(axis=1)[:, None], (detail_last - detail_first).prod(axis=1))
    overlaps = common >= .36 * np.maximum(1, smaller)
    matches_per_original = overlaps.sum(axis=1)

    revised = original.copy()
    insertions = [[] for _ in range(len(original) + 1)]
    added = respaced = 0
    for index, line in enumerate(detailed):
        linked = np.flatnonzero(overlaps[:, index])
        label = line['text']
        if len(linked) == 0 and line['score'] >= .90:
            letters_or_digits = [char for char in label if char.isalnum()]
            if len(letters_or_digits) >= 2 or any(char.isdigit() for char in letters_or_digits):
                x, y = detail_first[index]
                following = np.flatnonzero((first[:, 1] > y + 8) | ((abs(first[:, 1] - y) <= 8) & (first[:, 0] > x)))
                at = following[0] if len(following) else len(original)
                insertions[at].append(line)
                added += 1
        elif len(linked) == 1 and matches_per_original[linked[0]] == 1 and line['score'] >= .92:
            previous = original[linked[0]]['text']
            # The second pass may restore spacing, but it cannot substitute a
            # conflicting number, sign, comparator, or unit for a known line.
            if _restores_word_boundaries(previous, label):
                revised[linked[0]] = line
                respaced += 1

    merged = []
    for index, line in enumerate(revised):
        merged.extend(insertions[index])
        merged.append(line)
    merged.extend(insertions[-1])
    return merged, added, respaced


def refine_export_text(image, ocr, cache_dir):
    """Recover small labels on selected native exports without changing grouping OCR."""
    if ocr.get('status') != 'ok' or not _export_text_needs_higher_resolution(image, ocr.get('lines', [])):
        return ocr
    metadata = {'detector_long_edge': 1536, 'applied': False, 'added_lines': 0, 'respaced_lines': 0}
    try:
        detailed = _read_text(image, cache_dir, HIGH_RES_ENGINE_VERSION, lambda: engine(high_res=True))
    except Exception:
        # A failed optional reread still exports the original OCR result.
        return {**ocr, 'export_refinement': {**metadata, 'status': 'unavailable'}}
    lines, added, respaced = _merge_export_lines(ocr['lines'], detailed.get('lines', []))
    applied = bool(added or respaced)
    result = {**ocr, **_text_result(lines, ocr.get('engine', ENGINE_VERSION))} if applied else ocr
    return {**result, 'export_refinement': {**metadata, 'applied': applied, 'added_lines': added, 'respaced_lines': respaced}}


def normalized(text):
    return ' '.join(re.findall(r'[\w]+',unicodedata.normalize('NFKC',text).casefold()))


def numeric_text(text):
    # OCR sometimes reads a citation's leading digit twice, once as superscript.
    # Restrict correction to a citation-like range attached to a word.
    def citation(match):
        a,b,c=match.groups()
        return a+c if unicodedata.normalize('NFKC',b)==a else match.group(0)
    text=re.sub(r'(?<=[A-Za-z])(\d)([\u00b2\u00b3\u00b9\u2070-\u2079])([-\u2013]\d)',citation,text)
    text=_BIBLIOGRAPHY_PAGES.sub(r'\1–', text)
    text=unicodedata.normalize('NFKC',text)
    # Zero/O ambiguity inside a word is not strong evidence of a changed value.
    return re.sub(r'(?<=[A-Za-z])0(?=[A-Za-z]|\b)','o',text).casefold()


def _values(text):
    # The comma after an integer ratio separates two fields: 17/24,70.8%.
    # Do not treat all commas as separators since they can be decimal marks.
    text = _COUNT_THEN_PERCENT.sub(r'\1 ', text)
    return [value.replace('−', '-').replace(' ', '').lstrip('+')
            for value in _SIGNED_NUMBER.findall(text)]


def signature(item):
    if '_text_signature' in item:
        return item['_text_signature']
    h=item['slide_height']
    lines=[l for l in item.get('ocr',{}).get('lines',[]) if l['score']>=.80
           and np.mean(np.asarray(l['box'])[:,1])<h*.94]
    # Poll countdowns are changing interface chrome, not new clinical data.
    # Require both polling language and a circular timer around the number.
    poll=any(word in normalized(item.get('ocr',{}).get('text','')) for word in ('polling','poll ','vote','voting','recommend'))
    if poll and 'gray' in item:
        circles=cv2.HoughCircles(item['gray'],cv2.HOUGH_GRADIENT,1.2,40,param1=100,param2=25,minRadius=12,maxRadius=75)
        def timer(line):
            txt=normalized(line['text'])
            if not re.fullmatch(r'\d{1,3}(?: seconds?)?',txt) or circles is None:
                return False
            center=np.asarray(line['box']).mean(axis=0)/[item['slide_width'],h]*[480,270]
            return center[0]>240 and any(np.linalg.norm(center-c[:2])<c[2]*.6 for c in circles[0])
        lines=[line for line in lines if not timer(line)]
    tokens=Counter(normalized(' '.join(l['text'] for l in lines)).split())
    headings=[l for l in lines if np.mean(np.asarray(l['box'])[:,1])<h*.22
              and np.mean(np.asarray(l['box'])[:,0])<item['slide_width']*.80
              and len(normalized(l['text']))>12]
    headings=sorted(headings,key=lambda l:np.min(np.asarray(l['box'])[:,1]))[:2]
    if len(headings)==2:
        first,second=headings
        gap=float(np.mean(np.asarray(second['box'])[:,1])-np.mean(np.asarray(first['box'])[:,1]))
        if gap>1.6*max(first.get('height',20),second.get('height',20)):
            headings=headings[:1] # Don't dilute a unique title with a shared body sentence.
    rich=[]
    for line in lines:
        text=numeric_text(line['text'])
        numbers=_values(text)
        # A decimal and an integer can occupy the same cell. Preserve the
        # existing template structure for ranges and other multi-number lines.
        template=(' '.join(re.findall(r'\w+|#', _NUMBER.sub('#', text))) if len(numbers)==1
                  else _NUMBER.sub('#', normalized(text)))
        box = np.asarray(line['box']) / [item['slide_width'], item['slide_height']]
        rich.append({**line, 'center': box.mean(axis=0), 'size': np.ptp(box, axis=0),
                     'numbers':numbers, 'template':template})
    item['_text_signature']=(tokens,normalized(' '.join(l['text'] for l in headings)),rich)
    return item['_text_signature']


def _numeric_lines(item, lines):
    min_footer_height = item.get('ocr', {}).get('median_text_height', 20) * .70
    return [line for line in lines if line['score'] >= .92 and line['numbers']
            and (line['center'][1] <= .80 or line.get('height', 0) >= min_footer_height)]


def _matched_numeric_lines(a, b):
    # Find corresponding table cells before interpreting their values. An already
    # matching cell must compete for its partner with the neighboring rows.
    nearest_a, nearest_b = {}, {}
    for i, line_a in enumerate(a):
        for j, line_b in enumerate(b):
            dx, dy = np.abs(line_a['center'] - line_b['center'])
            width, height = np.minimum(line_a['size'], line_b['size'])
            if dx > min(.025, .65 * width) or dy > min(.025, .55 * height):
                continue
            if SequenceMatcher(None, line_a['template'], line_b['template']).ratio() <= .90:
                continue
            distance = dx * dx + dy * dy
            nearest_a[i] = min(nearest_a.get(i, (distance, j)), (distance, j))
            nearest_b[j] = min(nearest_b.get(j, (distance, i)), (distance, i))
    for i, (_, j) in nearest_a.items():
        if nearest_b[j][1] == i and a[i]['numbers'] != b[j]['numbers']:
            yield a[i], b[j]


def _numeric_box(shape, a, b, pad):
    height, width = shape[:2]
    corners = np.concatenate((a['box'], b['box']))
    x1, y1 = np.maximum(np.floor(corners.min(axis=0)).astype(int) - pad, 0)
    x2, y2 = np.minimum(np.ceil(corners.max(axis=0)).astype(int) + pad, [width, height])
    return x1, y1, x2, y2


def _compact_stroke(signed):
    return any(cv2.erode((polarity >= 11).astype(np.uint8), np.ones((2, 2), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0).any()
               for polarity in (signed, -signed))


def _same_numeric_pixels(left, right, a, b):
    if left.shape != right.shape:
        return False
    x1, y1, x2, y2 = _numeric_box(left.shape, a, b, 2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return False
    first, second = left[y1:y2, x1:x2], right[y1:y2, x1:x2]
    if cv2.absdiff(first, second).max() > 34:
        return False
    levels = [cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.int16) for patch in (first, second)]
    signed = levels[0] - levels[1]
    if np.abs(signed).max() > 26 or np.abs(signed).mean() > 4.3:
        return False
    # JPEG fringes can create connected differences where opposite brightness
    # changes meet. Preserve a coherent change of even a faint 2x2 glyph.
    return not _compact_stroke(signed)


def _same_reference_pixels(left, right, a, b):
    """Check native pixels when OCR disagrees about an inline reference."""
    if left.shape != right.shape:
        return False
    texts = a['text'], b['text']
    citation = all(_CITATION.search(text) for text in texts)
    first, second = (_FUSED_CITATION.search(text) for text in texts)
    if not citation and first and second:
        citation = (first[0] != second[0] and first[2] != second[2]
                    and first[1] == second[1] and first[3] == second[3]
                    and texts[0][:first.start()] == texts[1][:second.start()])
    affiliation = all(text.count(',') >= 3 and (re.search(r'[¹²³⁴⁵⁶⁷⁸⁹⁰]', text) or
                      len(re.findall(r',\s*\d{1,2}\b', text)) >= 3 or
                      (_AFFILIATION.search(text) and
                       (text.count(';') >= 2 or _AFFILIATION_MARKER.search(text)))) for text in texts)
    # Use the relaxed antialiasing check only for a change in the reference
    # markers themselves. Body digits and affiliation names stay protected.
    affiliation_markers = affiliation and (
        _AFFILIATION_MARKER.sub(r'\1', texts[0]) == _AFFILIATION_MARKER.sub(r'\1', texts[1])
    )
    references = all(_REFERENCE_TEXT.search(text) and _REFERENCE_LIST.search(text) for text in texts)
    if not (citation or affiliation or references):
        return False
    x1, y1, x2, y2 = _numeric_box(left.shape, a, b, 2)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return False
    patches = [cv2.GaussianBlur(image[y1:y2, x1:x2], (3, 3), 0) for image in (left, right)]
    gray_a, gray_b = (cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) for patch in patches)
    delta = cv2.absdiff(gray_a, gray_b)
    core = cv2.erode((delta > 12).astype(np.uint8), np.ones((2, 2), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)
    if delta.max() <= 20 and not core.any() and (citation or affiliation):
        return True
    # A numbered reference list can cause a nearby marker or punctuation to
    # be read as a value. Require unusually close agreement for the whole line.
    if references and delta.max() <= 9 and cv2.absdiff(*patches).max() <= 16:
        return True
    if not affiliation_markers or not (delta.max() <= 32 and delta.mean() <= 2.2):
        return False
    # A stable dark stroke may be sharpened differently from one JPEG to the
    # next. Test whether the residual still follows the same stroke on both.
    active = delta > 12
    dilated_a, dilated_b = (cv2.dilate(gray, np.ones((3, 3), np.uint8)) for gray in (gray_a, gray_b))
    pale = np.minimum(dilated_a, dilated_b)
    dark = np.maximum(gray_a, gray_b)
    unsupported = active & (pale.astype(np.int16) - dark < 8)
    unsupported_core = cv2.erode(unsupported.astype(np.uint8), np.ones((2, 2), np.uint8),
                                 borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return (np.count_nonzero(active) <= 30 and np.count_nonzero(unsupported) <= 4
            and not unsupported_core.any())


def _white_arrow_on_dark(stroke, covered, clear):
    if np.median(clear) > 135:
        return False
    height, width = stroke.shape
    bright = (stroke > 65) & (covered >= 205)
    edge_width = max(1, round(.18*width))
    vertical_outline = np.count_nonzero((stroke[:, :edge_width] < -22).any(axis=1))
    right_edges = [np.flatnonzero(row)[-1] for row in bright[:round(.68*height)] if row.any()]
    # The arrow widens diagonally from its tip and leaves a separate lower tail.
    return (vertical_outline >= .62*height and len(right_edges) >= .43*height
            and max(right_edges[-2:])-min(right_edges[:2]) >= .30*width
            and np.count_nonzero(bright[round(.72*height):, width//2:]) >= 2)


def _cursor_bounds(covered, clear, first, second):
    if covered.shape != clear.shape:
        return None
    x1, y1, x2, y2 = _numeric_box(covered.shape, first, second, 5)
    if x2 - x1 < 6 or y2 - y1 < 9:
        return None
    levels = [cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.int16)
              for image in (covered, clear)]
    signed = levels[0] - levels[1]
    mask = (np.abs(signed) >= 30).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for x, y, w, h, count in stats[1:]:
        if not (6 <= w <= 22 and 9 <= h <= 30 and .42 <= w / h <= .94):
            continue
        if x > .40 * signed.shape[1] or cv2.countNonZero(mask) - count > 8:
            continue
        # The outline is visible even when the white interior is on white paper.
        # Removing a printed digit cannot add the separate outline and tail.
        stroke = signed[y:y+h, x:x+w]
        left_edge = stroke[:int(.88 * h), :max(1, round(.18 * w))]
        middle = np.s_[:int(.65 * h), max(1, round(.14 * w)):round(.7 * w)]
        interior = stroke[middle]
        pale = levels[0][y:y+h, x:x+w][middle] >= 205
        tip = stroke[int(.72 * h):, int(.42 * w):]
        bright_fill = (np.count_nonzero(interior > 65) >= 4 or np.mean(pale) >= .58
                       or np.mean(pale[-3:]) >= .72)
        if (np.count_nonzero((left_edge < -65).any(axis=1)) >= .5 * h
                and bright_fill and np.count_nonzero(tip < -50) >= 4):
            return x1 + x, y1 + y, w, h
        position = np.s_[y:y+h, x:x+w]
        if _white_arrow_on_dark(stroke, levels[0][position], levels[1][position]):
            return x1 + x, y1 + y, w, h
    return None


def _cursor_occludes(covered, clear, shorter, longer):
    before, after = shorter['numbers'], longer['numbers']
    removed_token = len(after) == len(before) + 1 and len(after[0]) <= 2 and after[1:] == before
    shortened = (len(before) == len(after) and sum(x != y for x, y in zip(before, after)) == 1
                 and any(0 < len(y)-len(x) <= 2 and (y.endswith(x) or y.startswith(x))
                         for x, y in zip(before, after)))
    return bool((removed_token or shortened) and _cursor_bounds(covered, clear, shorter, longer))


def _cursor_omits_decimal(covered, clear, first, second):
    """Require the physical decimal glyph to survive outside a fitted pointer."""
    before, after = first['numbers'][0], second['numbers'][0]
    if not re.fullmatch(r'-?\d+\.\d+', after) or before != after.replace('.', ''):
        return False
    cursor = _cursor_bounds(covered, clear, first, second)
    if cursor is None:
        return False
    x, y, width, height = cursor
    x1, y1, x2, y2 = _numeric_box(clear.shape, first, second, 5)
    y2 = min(clear.shape[0], max(y2, y + 2 * height))
    x, y = x - x1, y - y1
    a, b = (cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY) for image in (covered, clear))
    delta = cv2.absdiff(a, b)
    outside = delta.copy()
    outside[max(0, y - 2):y + height + 2, max(0, x - 2):x + width + 2] = 0
    if outside.max() > 21 or _compact_stroke(outside.astype(np.int16)):
        return False
    ink = (b < min(135, np.median(b) - 60)).astype(np.uint8)
    _, _, components, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    for dx, dy, dw, dh, area in components[1:]:
        if (not (1 <= dw <= 4 and 1 <= dh <= 4 and area >= 3)
                or abs(dx - x) > max(3, width // 2)
                or not y + height + 2 <= dy <= y + height + 10
                or delta[dy:dy + dh, dx:dx + dw].max() > 10):
            continue
        baseline = dy + dh
        nearby_digits = [px for px, py, pw, ph, _ in components[1:]
                         if ph >= height * .65 and abs(py + ph - baseline) <= 2
                         and abs(px - dx) <= height]
        if any(px < dx for px in nearby_digits) and any(px > dx for px in nearby_digits):
            return True
    return False


def _recognize_cell(patch):
    height, width = patch.shape[:2]
    scale = min(4, max(.5, min(64 / height, 1000 / width)))
    if scale != 1:
        patch = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    result = engine()(patch, use_det=False, use_cls=False, use_rec=True, text_score=.75)
    if result.txts:
        return result.txts[0], float(result.scores[0])
    return '', 0.


def _read_numeric_patch(patch):
    try:
        return _recognize_cell(patch)
    except (cv2.error, OSError, RuntimeError, ValueError):
        return '', 0.


def _numeric_review(bounds_a, bounds_b, text_a, text_b, read_a, read_b):
    return {'bounds_a': list(map(int, bounds_a)), 'bounds_b': list(map(int, bounds_b)),
            'text_a': text_a, 'text_b': text_b, 'read_a': read_a[0], 'read_b': read_b[0],
            'confidence_a': round(read_a[1], 4), 'confidence_b': round(read_b[1], 4)}


def _reread_numeric_line(left, right, a, b):
    if left.shape != right.shape:
        return False, None
    x1, y1, x2, y2 = bounds = _numeric_box(left.shape, a, b, 3)
    if not (5 <= y2 - y1 <= left.shape[0] * .10 and 7 <= x2 - x1 <= left.shape[1] * .55):
        return False, None
    patches = [image[y1:y2, x1:x2] for image in (left, right)]
    reads = [_read_numeric_patch(patch) for patch in patches]
    values = [_values(numeric_text(read[0])) for read in reads]
    strong = all(values) and min(read[1] for read in reads) >= .94
    if strong and values == [a['numbers'], b['numbers']]:
        return False, None
    if strong and values[0] == values[1] and normalized(reads[0][0]) == normalized(reads[1][0]):
        gray = [cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) for patch in patches]
        signed = np.subtract(*gray, dtype=np.int16)
        blurred = [cv2.GaussianBlur(image, (3, 3), 0) for image in gray]
        delta = cv2.absdiff(*blurred)
        if delta.max() <= 22 and not _compact_stroke(signed):
            return True, None
    return False, _numeric_review(bounds, bounds, a['text'], b['text'], *reads)


def local_numeric_evidence(left, right, *, occlusions=None, unresolved=None):
    """Reread isolated glyph changes that full-slide OCR could miss."""
    if left.shape != right.shape:
        return []
    gray_a, gray_b = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in (left, right))
    mask = (cv2.absdiff(gray_a, gray_b) >= 34).astype(np.uint8)
    changed = cv2.countNonZero(mask)
    height, width = mask.shape
    if not 6 <= changed <= min(260, mask.size * .0004):
        return []
    glyph_height = max(5, round(height * .012))
    joined = cv2.dilate(mask, np.ones((5, 7), np.uint8))
    _, _, regions, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    if len(regions) > 5:
        return []
    for x, y, w, h, _ in regions[1:]:
        if not (7 <= w <= glyph_height * 4 and 5 <= h <= glyph_height * 2.4):
            continue
        if cv2.countNonZero(mask[y:y+h, x:x+w]) < 6:
            continue
        x1, x2 = max(0, x - glyph_height), min(width, x + w + glyph_height)
        y1, y2 = max(0, y - 4), min(height, y + h + 4)
        a, b = (_read_numeric_patch(image[y1:y2, x1:x2]) for image in (left, right))
        confidence = min(a[1], b[1])
        if confidence < (.78 if unresolved is not None else .94):
            continue
        if not all(_SIGNED_NUMBER.fullmatch(numeric_text(value)) for value, _ in (a, b)):
            continue
        a_values, b_values = (_values(numeric_text(value[0])) for value in (a, b))
        if a_values == b_values:
            continue
        box = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
        first, second = ({'box': box, 'numbers': values} for values in (a_values, b_values))
        if _cursor_occludes(left, right, first, second):
            if occlusions is not None:
                occlusions.append('a')
            continue
        if _cursor_occludes(right, left, second, first):
            if occlusions is not None:
                occlusions.append('b')
            continue
        if confidence < .94:
            reads, lines, images = (a, b), (first, second), (left, right)
            for index, other in ((0, 1), (1, 0)):
                if (reads[index][1] < .94 <= reads[other][1]
                        and _cursor_omits_decimal(images[index], images[other], lines[index], lines[other])):
                    if occlusions is not None:
                        occlusions.append('ab'[index])
                    break
            else:
                bounds = (x1, y1, x2, y2)
                unresolved.append(_numeric_review(bounds, bounds, '', '', a, b))
            continue
        return [[a[0], b[0]]]
    return []


def text_evidence(a,b,*,native_pair=None,reread_numeric=False):
    ta,ha,la=signature(a); tb,hb,lb=signature(b)
    shared=sum((ta&tb).values())
    ca=shared/max(1,sum(ta.values())); cb=shared/max(1,sum(tb.values()))
    heading=SequenceMatcher(None,ha,hb).ratio() if ha and hb else 0.
    numeric_changes, numeric_occlusions, numeric_reviews = [], [], []
    rereads = 0
    if max(ca,cb)>.65 or heading>.80:
        images = None
        for line_a, line_b in _matched_numeric_lines(_numeric_lines(a, la), _numeric_lines(b, lb)):
            if native_pair is not None and images is None:
                images = native_pair()
            if images is not None:
                left, right = images
                if (_same_numeric_pixels(left, right, line_a, line_b) or
                        _same_reference_pixels(left, right, line_a, line_b)):
                    continue
                if _cursor_occludes(left, right, line_a, line_b):
                    numeric_occlusions.append('a')
                    continue
                if _cursor_occludes(right, left, line_b, line_a):
                    numeric_occlusions.append('b')
                    continue
                if reread_numeric and rereads < 3:
                    rereads += 1
                    same, review = _reread_numeric_line(left, right, line_a, line_b)
                    if same:
                        continue
                    if review is not None:
                        numeric_reviews.append(review)
            numeric_changes.append([line_a['text'], line_b['text']])
            if not reread_numeric or rereads >= 3 or len(numeric_changes) >= 3:
                break
    return {'text_cover_a':round(ca,4),'text_cover_b':round(cb,4),
            'heading_similarity':round(heading,4),'numeric_conflict':bool(numeric_changes),
            'numeric_changes':numeric_changes,
            'numeric_reviews':numeric_reviews,
            'numeric_occlusions': sorted(set(numeric_occlusions)),
            'text_words_a':sum(ta.values()),'text_words_b':sum(tb.values())}
