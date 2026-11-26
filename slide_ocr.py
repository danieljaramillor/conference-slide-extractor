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
_local=threading.local()
_NUMBER = re.compile(r'\d+(?:[.,]\d+)?')
_SIGNED_NUMBER = re.compile(r'(?<![\w.,])[-+−]\s*\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?')
_COUNT_THEN_PERCENT = re.compile(r'(\d+\s*/\s*\d+)\s*,\s*(?=\d+(?:[.,]\d+)?\s*%)')
_BIBLIOGRAPHY_PAGES = re.compile(r'(\b(?:19|20)\d{2}\s*[:;, ]\s*\d+\s*:\s*\d+)\s*–\s*-\s*(?=\d+\b)')
_CITATION = re.compile(r'(?<=\w)[\{\^]\d{1,4}\s*$')
_AFFILIATION = re.compile(r'\b(?:hospital|university|institute|department|college|cent(?:er|re)|oncology)\b', re.I)


def engine():
    if not hasattr(_local,'ocr'):
        import rapidocr
        from rapidocr import RapidOCR
        models=Path(rapidocr.__file__).parent/'models'
        paths={'Det.model_path':models/'PP-OCRv6_det_small.onnx',
               'Rec.model_path':models/'PP-OCRv6_rec_small.onnx',
               'Cls.model_path':models/'ch_ppocr_mobile_v2.0_cls_mobile.onnx'}
        for path in paths.values():
            if not path.is_file():
                raise RuntimeError(f'Missing bundled OCR model {path}. Reinstall rapidocr==3.10.0.')
        _local.ocr=RapidOCR(params={**{k:str(v) for k,v in paths.items()},
            'EngineConfig.onnxruntime.intra_op_num_threads':1,
            'EngineConfig.onnxruntime.inter_op_num_threads':1,
            'Det.limit_side_len':960,'Det.limit_type':'max',
            'Global.max_side_len':4096,'Global.log_level':'error'})
    return _local.ocr


def read_text(image, cache_dir):
    key=hashlib.sha256(ENGINE_VERSION.encode()+str(image.shape).encode()+image.tobytes()).hexdigest()
    cache_dir=Path(cache_dir)
    cache_dir.mkdir(parents=True,exist_ok=True)
    path=cache_dir/f'{key}.json'
    if path.exists():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (OSError,ValueError):
            pass
    result=engine()(image,use_det=True,use_cls=False,use_rec=True,text_score=.50)
    lines=[]
    if result.txts is not None:
        for text,score,box in zip(result.txts,result.scores,result.boxes):
            pts=np.asarray(box,float)
            lines.append({'text':str(text),'score':round(float(score),4),'box':pts.tolist(),
                          'height':round(float((np.linalg.norm(pts[3]-pts[0])+np.linalg.norm(pts[2]-pts[1]))/2),2)})
    strong=[line for line in lines if line['score']>=.75]
    output={'status':'ok','engine':ENGINE_VERSION,'lines':lines,
            'text':'\n'.join(line['text'] for line in lines),
            'characters':sum(len(line['text']) for line in strong),
            'mean_confidence':round(float(np.mean([line['score'] for line in strong])),3) if strong else 0.,
            'median_text_height':round(float(np.median([line['height'] for line in strong])),2) if strong else 0.}
    # Each thread writes a different temporary filename, then atomically replaces.
    temp=path.with_suffix(f'.{uuid.uuid4().hex}.tmp')
    temp.write_text(json.dumps(output,ensure_ascii=False),encoding='utf-8')
    temp.replace(path)
    return output


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
    for polarity in (signed, -signed):
        ink = (polarity >= 11).astype(np.uint8)
        if cv2.erode(ink, np.ones((2, 2), np.uint8), borderType=cv2.BORDER_CONSTANT,
                     borderValue=0).any():
            return False
    return True


def _same_reference_pixels(left, right, a, b):
    """Allow compression noise in reference markers after checking their ink."""
    texts = a['text'], b['text']
    citation = all(_CITATION.search(text) for text in texts)
    affiliation = all(text.count(',') >= 3 and (re.search(r'[¹²³⁴⁵⁶⁷⁸⁹⁰]', text) or
                      len(re.findall(r',\s*\d{1,2}\b', text)) >= 3 or
                      (text.count(';') >= 2 and _AFFILIATION.search(text))) for text in texts)
    if left.shape != right.shape or not (citation or affiliation):
        return False
    x1, y1, x2, y2 = _numeric_box(left.shape, a, b, 2)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return False
    patches = [cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY) for image in (left, right)]
    delta = cv2.absdiff(*(cv2.GaussianBlur(patch, (3, 3), 0) for patch in patches))
    # A compact surviving stroke can be a real reference amendment.
    core = cv2.erode((delta > 12).astype(np.uint8), np.ones((2, 2), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return delta.max() <= 20 and not core.any()


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


def _cursor_occludes(covered, clear, shorter, longer):
    if covered.shape != clear.shape:
        return False
    before, after = shorter['numbers'], longer['numbers']
    removed_token = len(after) == len(before) + 1 and len(after[0]) <= 2 and after[1:] == before
    shortened = (len(before) == len(after) and sum(x != y for x, y in zip(before, after)) == 1
                 and any(0 < len(y)-len(x) <= 2 and (y.endswith(x) or y.startswith(x))
                         for x, y in zip(before, after)))
    if not (removed_token or shortened):
        return False
    x1, y1, x2, y2 = _numeric_box(covered.shape, shorter, longer, 5)
    if x2 - x1 < 6 or y2 - y1 < 9:
        return False
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
            return True
        position = np.s_[y:y+h, x:x+w]
        if _white_arrow_on_dark(stroke, levels[0][position], levels[1][position]):
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


def local_numeric_evidence(left, right, *, occlusions=None):
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
        a, b = (_recognize_cell(image[y1:y2, x1:x2]) for image in (left, right))
        if min(a[1], b[1]) < .94:
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
        return [[a[0], b[0]]]
    return []


def text_evidence(a,b,*,native_pair=None):
    ta,ha,la=signature(a); tb,hb,lb=signature(b)
    shared=sum((ta&tb).values())
    ca=shared/max(1,sum(ta.values())); cb=shared/max(1,sum(tb.values()))
    heading=SequenceMatcher(None,ha,hb).ratio() if ha and hb else 0.
    numeric_changes, numeric_occlusions = [], []
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
            numeric_changes.append([line_a['text'], line_b['text']])
            break
    return {'text_cover_a':round(ca,4),'text_cover_b':round(cb,4),
            'heading_similarity':round(heading,4),'numeric_conflict':bool(numeric_changes),
            'numeric_changes':numeric_changes,
            'numeric_occlusions': sorted(set(numeric_occlusions)),
            'text_words_a':sum(ta.values()),'text_words_b':sum(tb.values())}
