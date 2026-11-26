"""Per-image slide regions, with native-resolution extraction and quality metadata."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def order_quad(points):
    pts = np.asarray(points, np.float32).reshape(4, 2)
    # Angle sorting avoids duplicate corners with diamond-shaped quadrilaterals.
    center = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1]-center[1], pts[:, 0]-center[0]))]
    return np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)


def extract(image, region):
    h, w = image.shape[:2]
    if region.get('quad') is None:
        x1, y1, x2, y2 = region['crop']
        return image[round(y1*h):round(y2*h), round(x1*w):round(x2*w)]
    pts = np.asarray(region['quad'], np.float32) * [w, h]
    tl, tr, br, bl = pts
    # Use the smaller projected edge: rectification never invents extra resolution.
    width = max(2, int(round(min(np.linalg.norm(tr-tl), np.linalg.norm(br-bl)))))
    height = max(2, int(round(min(np.linalg.norm(bl-tl), np.linalg.norm(br-tr)))))
    target = np.float32([[0,0], [width-1,0], [width-1,height-1], [0,height-1]])
    transform = cv2.getPerspectiveTransform(pts.astype(np.float32), target)
    return cv2.warpPerspective(image, transform, (width,height), flags=cv2.INTER_LINEAR)


def _text_strokes(gray):
    edge = cv2.Canny(gray, 60, 140)
    n, _, stats, _ = cv2.connectedComponentsWithStats((edge > 0).astype(np.uint8), 8)
    if n < 2:
        return 0
    stats = stats[1:]
    return int(np.sum((stats[:,2] >= 2) & (stats[:,2] <= 45) &
                      (stats[:,3] >= 3) & (stats[:,3] <= 24) & (stats[:,4] >= 4)))


def _refine_rectangle(image,box,radius):
    """Recover native pixel boundaries lost in the small detection image."""
    h,w=image.shape[:2]
    x1,y1,x2,y2=[round(v*d) for v,d in zip(box,(w,h,w,h))]
    def vertical(x):
        lo,hi=max(0,x-radius),min(w,x+radius+1)
        strip=image[max(0,y1+4):min(h,y2-4),lo:hi].astype(np.float32)
        if strip.shape[0]<2 or strip.shape[1]<2:
            return x
        values=np.abs(np.diff(strip,axis=1)).mean(axis=(0,2))
        return lo+int(np.argmax(values))+1 if float(values.max())>15 else x
    x1,x2=vertical(x1),vertical(x2)
    def horizontal(y):
        lo,hi=max(0,y-radius),min(h,y+radius+1)
        strip=image[lo:hi,max(0,x1+4):min(w,x2-4)].astype(np.float32)
        if strip.shape[0]<2 or strip.shape[1]<2:
            return y
        values=np.abs(np.diff(strip,axis=0)).mean(axis=(1,2))
        return lo+int(np.argmax(values))+1 if float(values.max())>15 else y
    y1,y2=horizontal(y1),horizontal(y2)
    return (x1/w,y1/h,x2/w,y2/h)


@dataclass(slots=True)
class _PresenterLayout:
    crop: tuple[float, float, float, float]
    chrome: np.ndarray
    slide: np.ndarray
    camera: np.ndarray


def _bright_panel(gray, threshold):
    mask = (gray > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    if cv2.contourArea(contour) < w * h * .78:
        return None
    return (x, y, x + w, y + h)


def _uniform_rows(image):
    colors = np.median(image, axis=1)[:, None]
    return np.mean(np.max(np.abs(image - colors), axis=2) < 17)


def broadcast_crop_profile(image, crop):
    """Verify an already observed broadcast crop against this frame's edges."""
    height, width = image.shape[:2]
    if width < 320 or height < 180:
        return None
    sw = min(width, 480)
    sh = round(height * sw / width)
    x1, y1, x2, y2 = [round(value*size) for value, size in zip(crop, (sw, sh, sw, sh))]
    w, h = x2-x1, y2-y1
    if not (.54 <= w/sw <= .80 and .60 <= h/sh <= .87 and
            .05 <= y1/sh <= .24 and .76 <= y2/sh <= .95 and 1.30 <= w/h <= 2.12):
        return None
    inset_on_left = x1 <= .085*sw and sw-x2 >= .17*sw
    inset_on_right = sw-x2 <= .085*sw and x1 >= .17*sw
    if not (inset_on_left or inset_on_right):
        return None
    small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
    header, footer = small[:y1-2], small[y2+2:]
    side = small[y1+round(h*.86):y2-2, x2+6:-6] if inset_on_left else small[y1+round(h*.86):y2-2, 6:x1-6]
    header_color, footer_color, side_color = (np.median(band, axis=(0, 1)) for band in (header, footer, side))
    if (np.ptp(header_color) < 17 or np.max(np.abs(header_color-footer_color)) < 24
            or np.max(np.abs(side_color-footer_color)) > 65
            or min(_uniform_rows(band) for band in (header, footer, side)) < .73):
        return None
    edge = x2 if inset_on_left else x1
    stripes = ((small[y1+2,x1+8:x2-8], small[y1-3,x1+8:x2-8]),
               (small[y2-3,x1+8:x2-8], small[y2+2,x1+8:x2-8]),
               (small[y1+8:y2-8,edge-3], small[y1+8:y2-8,edge+2]))
    if any(np.mean(cv2.absdiff(a,b).max(axis=1) >= 24) < .82 for a,b in stripes):
        return None
    return np.vstack([cv2.resize(band, (72, 12), interpolation=cv2.INTER_AREA) for band in (header, footer)])


def presenter_layout(image):
    """Collect evidence of a broadcast container, without changing the crop."""
    height, width = image.shape[:2]
    if width < 320 or height < 180 or not 1.35 <= width / height <= 2.4:
        return None
    sw = min(480, width)
    sh = round(height * sw / width)
    small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
    header_color = np.median(small[:max(1, round(sh * .04))], axis=(0, 1))
    if np.ptp(header_color) < 17 or np.mean(header_color) > 180:
        return None

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    panel = _bright_panel(gray, 210)
    core = _bright_panel(gray, 235)
    if panel is None or core is None or max(abs(a - b) for a, b in zip(panel, core)) > 3:
        return None
    x1, y1, x2, y2 = panel
    pane_width, pane_height = x2 - x1, y2 - y1
    if not (.52 <= pane_width / sw <= .80 and .60 <= pane_height / sh <= .87 and
            .045 <= y1 / sh <= .24 and .76 <= y2 / sh <= .952 and
            1.30 <= pane_width / pane_height <= 2.12):
        return None

    margin = max(2, round(sw * .013))
    if x1 <= sw * .085 and sw - x2 >= sw * .17:
        sx1, sx2 = x2 + margin, sw - margin
    elif sw - x2 <= sw * .085 and x1 >= sw * .17:
        sx1, sx2 = margin, x1 - margin
    else:
        return None

    header = small[:y1 - 2]
    footer = small[y2 + 2:]
    lower_side = small[y1 + round(pane_height * .86):y2 - 2, sx1:sx2]
    if min(_uniform_rows(header), _uniform_rows(footer), _uniform_rows(lower_side)) < .73:
        return None
    side_color = np.median(lower_side, axis=(0, 1))
    footer_color = np.median(footer, axis=(0, 1))
    if (np.max(np.abs(side_color - footer_color)) > 65 or
            np.max(np.abs(header_color - footer_color)) < 24):
        return None

    camera = gray[y1 + round(pane_height * .04):y1 + round(pane_height * .38), sx1:sx2]
    # A bright independent content panel can have its own build. Do not
    # infer from its motion that it is a camera feed.
    if np.mean(camera > 210) > .30 or np.ptp(np.quantile(camera, (.10, .90))) < 31:
        return None
    slide = gray[y1:y2, x1:x2]
    if _text_strokes(slide) < 8:
        return None

    normalized = (x1 / sw, y1 / sh, x2 / sw, y2 / sh)
    crop = _refine_rectangle(image, normalized, max(4, round(2 * width / sw)))
    chrome = np.vstack([cv2.resize(band, (72, 12), interpolation=cv2.INTER_AREA)
                        for band in (header, footer)])
    return _PresenterLayout(crop, chrome,
                            cv2.resize(slide, (144, 81), interpolation=cv2.INTER_AREA),
                            cv2.resize(camera, (72, 44), interpolation=cv2.INTER_AREA))


def _close_crop(first, second, tolerance):
    return max(abs(a - b) for a, b in zip(first, second)) <= tolerance


def _same_container(reference, candidate):
    return (_close_crop(reference.crop, candidate.crop, .012) and
            np.mean(cv2.absdiff(reference.chrome, candidate.chrome)) <= 4)


def _same_slide(first, second):
    delta = cv2.absdiff(first.slide, second.slide)
    return delta.mean() <= 1.1 and np.mean(delta > 12) <= .004


def _anchored_crop(layouts, known_crops, index):
    previous, current, following = layouts[index - 1:index + 2]
    before, after = known_crops[index - 1], known_crops[index + 1]
    if any(layout is None for layout in (previous, current, following)) or before is None or after is None:
        return None
    if (all(_same_container(current, neighbor) for neighbor in (previous, following)) and
            all(_close_crop(crop, current.crop, .005) for crop in (before, after)) and
            (_same_slide(previous, current) or _same_slide(current, following))):
        return current.crop
    return None


def confirmed_presenter_crops(layouts, known_crops=None):
    """Return crops supported by motion or by both adjacent trusted slide panes."""
    if known_crops is None:
        known_crops = [None] * len(layouts)
    else:
        layouts = [layout if crop is None or (layout is not None and _close_crop(crop, layout.crop, .005)) else None
                   for layout, crop in zip(layouts, known_crops, strict=True)]
    confirmed = {}
    start = 0
    for end, candidate in enumerate([*layouts, None]):
        if candidate is not None and _same_container(layouts[start], candidate):
            continue
        if end - start >= 3:
            motion = []
            for previous, current in zip(layouts[start:end - 1], layouts[start + 1:end]):
                if _same_slide(previous, current):
                    camera_delta = cv2.absdiff(previous.camera, current.camera)
                    if camera_delta.mean() >= 4:
                        motion.append(np.mean(camera_delta > 16))
            if sum(value >= .055 for value in motion) >= 2 and any(.055 <= value <= .55 for value in motion):
                confirmed.update((i, layouts[i].crop) for i in range(start, end) if known_crops[i] is None)
        start = end if candidate is not None else end + 1
    for index in range(1, len(layouts) - 1):
        if index not in confirmed and known_crops[index] is None:
            crop = _anchored_crop(layouts, known_crops, index)
            if crop is not None:
                confirmed[index] = crop
    return confirmed


def detect_region(image):
    h, w = image.shape[:2]
    scale = min(1., 800/max(h,w))
    small = cv2.resize(image, (max(2,round(w*scale)),max(2,round(h*scale))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    border = np.concatenate([small[:3].reshape(-1,3),small[-3:].reshape(-1,3),
                             small[:,:3].reshape(-1,3),small[:,-3:].reshape(-1,3)])
    bg = np.median(border,axis=0)
    border_consistency = float(np.mean(np.max(np.abs(border.astype(float)-bg),axis=1)<22))
    default = {'crop':(0.,0.,1.,1.), 'quad':None, 'crop_mode':'full',
               'crop_confidence': .93 if border_consistency>.8 else .65,
               'region_candidates':0, 'region_ambiguous':False}
    # A clean white/black slide border is stronger evidence for a full slide than
    # an internal chart's rectangle. It also prevents cutting off slide headings.
    neutral = max(bg)-min(bg)<20
    if border_consistency>.95 and neutral and (float(bg.mean())>225 or float(bg.mean())<24):
        # Letterboxes have a different center and long uniform bars; still inspect them.
        center = np.median(small[sh//3:2*sh//3,sw//3:2*sw//3].reshape(-1,3),axis=0)
        if float(bg.mean())>225 or np.max(np.abs(center-bg))<28:
            return default

    delta = np.max(np.abs(small.astype(float)-bg),axis=2)
    masks = [(delta>28).astype(np.uint8)*255]
    masks.extend([(gray>v).astype(np.uint8)*255 for v in (145,210)])
    edges = cv2.Canny(gray,50,120)
    masks.append(cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)))
    candidates=[]
    for mi, mask in enumerate(masks):
        if mi==0:
            mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:35]:
            area=cv2.contourArea(contour)/(sw*sh)
            if not .16<area<.96:
                continue
            perimeter=cv2.arcLength(contour,True)
            poly=cv2.approxPolyDP(contour,.018*perimeter,True)
            if len(poly)!=4 or not cv2.isContourConvex(poly):
                # Background masks often have rounded slide corners.
                if mi!=0:
                    continue
                x,y,cw,ch=cv2.boundingRect(contour)
                if cv2.contourArea(contour)/(cw*ch)<.90:
                    continue
                poly=np.float32([[x,y],[x+cw,y],[x+cw,y+ch],[x,y+ch]])
            pts=order_quad(poly)
            tl,tr,br,bl=pts
            ww=(np.linalg.norm(tr-tl)+np.linalg.norm(br-bl))/2
            hh=(np.linalg.norm(bl-tl)+np.linalg.norm(br-tr))/2
            ratio=ww/max(1,hh)
            if not .55<ratio<2.5:
                continue
            x,y,cw,ch=cv2.boundingRect(pts.astype(np.int32))
            if cw<sw*.20 or ch<sh*.20:
                continue
            # Reject a region whose corners severely collapse in perspective.
            if min(np.linalg.norm(tr-tl),np.linalg.norm(br-bl))/max(ww,1)<.60:
                continue
            sub=gray[max(0,y):min(sh,y+ch),max(0,x):min(sw,x+cw)]
            strokes=_text_strokes(sub)
            if strokes<8:
                continue
            rect_area=cw*ch
            rectangularity=cv2.contourArea(pts.astype(np.float32))/max(1,rect_area)
            aspect_prior=1. if 1.20<ratio<1.95 else .65
            score=.32*min(1,strokes/80)+.26*rectangularity+.22*min(1,area/.55)+.12*aspect_prior+.08*(mi==0)
            normalized=pts/np.float32([sw,sh])
            if any(np.max(np.abs(normalized-c['points']))<.018 for c in candidates):
                continue
            candidates.append({'points':normalized,'score':float(score),'area':float(area),
                               'rectangularity':float(rectangularity),'strokes':strokes,'origin':mi})
    if not candidates:
        return default
    candidates.sort(key=lambda c:c['score'],reverse=True)
    best=candidates[0]
    trusted_background=best['origin']==0 and border_consistency>.90
    if best['score']<.85 and not (trusted_background and best['score']>=.68):
        default['region_candidates']=len(candidates)
        default['crop_confidence']=.65
        return default
    pts=best['points']
    x1,y1=pts.min(axis=0); x2,y2=pts.max(axis=0)
    # Preserve native pixels for axis-aligned screenshots. Warping is only for
    # projected screens, and is reported separately because interpolation occurs.
    aligned=max(abs(pts[0,1]-pts[1,1]),abs(pts[2,1]-pts[3,1]),
                abs(pts[0,0]-pts[3,0]),abs(pts[1,0]-pts[2,0]))<.040
    ambiguity=any(c['score']>best['score']-.045 and
                  np.max(np.abs(c['points']-pts))>.10 for c in candidates[1:])
    box=tuple(float(v) for v in (x1,y1,min(1,x2),min(1,y2)))
    if aligned:
        box=_refine_rectangle(image,box,max(4,round(3/scale)))
    return {'crop':box,
            'quad':None if aligned else pts.tolist(),
            'crop_mode':'auto-inset' if aligned else 'auto-perspective',
            'crop_confidence':round(best['score'],3), 'region_candidates':len(candidates),
            'region_ambiguous':ambiguity}


def quality(image):
    h,w=image.shape[:2]
    gray=cv2.cvtColor(cv2.resize(image,(480,270),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
    return {'slide_width':w,'slide_height':h,'slide_pixels':w*h,
            'sharpness':float(cv2.Laplacian(cv2.GaussianBlur(gray,(3,3),0),cv2.CV_32F).var())}


def recover_projected_screen(crop, region):
    """Find a visible projection inside an inset that encloses a stage camera."""
    if region['crop_mode'] != 'auto-inset' or region.get('quad') is not None:
        return None
    # A dark venue or speaker-only inset is a candidate. Leave bright slides
    # and original full-frame slides to the normal detector.
    preview = cv2.cvtColor(cv2.resize(crop, (480, 270), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    if float(np.mean(preview < 50)) < .53:
        return None
    bright = (preview > 110).astype(np.uint8) * 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
        area = cv2.contourArea(contour) / preview.size
        if not .19 < area < .58:
            continue
        quad = cv2.approxPolyDP(contour, .025 * cv2.arcLength(contour, True), True)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            continue
        points = order_quad(quad)
        left, top = points.min(axis=0)
        right, bottom = points.max(axis=0)
        if left < 20 or right > 460 or top < 6 or bottom > 220:
            continue
        if abs(points[0, 1] - points[1, 1]) < 9:
            continue  # Require perspective evidence to avoid charts on dark slides.
        width = (np.linalg.norm(points[1] - points[0]) + np.linalg.norm(points[2] - points[3])) / 2
        height = (np.linalg.norm(points[3] - points[0]) + np.linalg.norm(points[2] - points[1])) / 2
        if not 1.18 < width / height < 2.25 or _text_strokes(preview[int(top):int(bottom), int(left):int(right)]) < 15:
            continue
        x1, y1, x2, y2 = region['crop']
        source_points = points / [480, 270] * [x2 - x1, y2 - y1] + [x1, y1]
        low, high = source_points.min(axis=0), source_points.max(axis=0)
        return {**region, 'crop': (*low.tolist(), *high.tolist()), 'quad': source_points.tolist(),
                'crop_mode': 'projected-screen', 'crop_confidence': .80}
    return None
