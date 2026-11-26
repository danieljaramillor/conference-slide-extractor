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
    result=engine()(image,use_cls=False)
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
    text=unicodedata.normalize('NFKC',text)
    # Zero/O ambiguity inside a word is not strong evidence of a changed value.
    return re.sub(r'(?<=[A-Za-z])0(?=[A-Za-z]|\b)','o',text).casefold()


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
        rich.append({**line,'center':np.asarray(line['box']).mean(axis=0)/[item['slide_width'],item['slide_height']],
                     'numbers':_NUMBER.findall(text),
                     'template':' '.join(re.findall(r'\w+|#', _NUMBER.sub('#', text)))})
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
            dx, dy = line_a['center'] - line_b['center']
            if max(abs(dx), abs(dy)) > .025:
                continue
            if SequenceMatcher(None, line_a['template'], line_b['template']).ratio() <= .90:
                continue
            distance = dx * dx + dy * dy
            nearest_a[i] = min(nearest_a.get(i, (distance, j)), (distance, j))
            nearest_b[j] = min(nearest_b.get(j, (distance, i)), (distance, i))
    for i, (_, j) in nearest_a.items():
        if nearest_b[j][1] == i and a[i]['numbers'] != b[j]['numbers']:
            yield a[i], b[j]


def _same_numeric_pixels(left, right, a, b):
    # An OCR segmentation error can change the transcription of an unchanged cell.
    # Clear it only when the native pixels already agree in both OCR boxes.
    if left.shape != right.shape:
        return False
    height, width = left.shape[:2]
    corners = np.concatenate((a['box'], b['box']))
    x1, y1 = np.maximum(np.floor(corners.min(axis=0)).astype(int) - 2, 0)
    x2, y2 = np.minimum(np.ceil(corners.max(axis=0)).astype(int) + 2, [width, height])
    if x2 - x1 < 3 or y2 - y1 < 3:
        return False
    delta = cv2.absdiff(left[y1:y2, x1:x2], right[y1:y2, x1:x2]).max(axis=2)
    if delta.max() > 16 or delta.mean() > 4:
        return False
    # JPEG noise can move individual pixels. A connected change in a glyph is
    # enough to preserve the OCR conflict even when the contrast is low.
    _, _, regions, _ = cv2.connectedComponentsWithStats((delta > 9).astype(np.uint8))
    return regions[1:, cv2.CC_STAT_AREA].max(initial=0) < 4


def text_evidence(a,b,*,native_pair=None):
    ta,ha,la=signature(a); tb,hb,lb=signature(b)
    shared=sum((ta&tb).values())
    ca=shared/max(1,sum(ta.values())); cb=shared/max(1,sum(tb.values()))
    heading=SequenceMatcher(None,ha,hb).ratio() if ha and hb else 0.
    numeric_changes=[]
    if max(ca,cb)>.65 or heading>.80:
        images = None
        for line_a, line_b in _matched_numeric_lines(_numeric_lines(a, la), _numeric_lines(b, lb)):
            if native_pair is not None and images is None:
                images = native_pair()
            if images is not None and _same_numeric_pixels(*images, line_a, line_b):
                continue
            numeric_changes.append([line_a['text'], line_b['text']])
            break
    return {'text_cover_a':round(ca,4),'text_cover_b':round(cb,4),
            'heading_similarity':round(heading,4),'numeric_conflict':bool(numeric_changes),
            'numeric_changes':numeric_changes,
            'text_words_a':sum(ta.values()),'text_words_b':sum(tb.values())}
