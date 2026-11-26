"""Per-image slide regions, with native-resolution extraction and quality metadata."""
from __future__ import annotations

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
