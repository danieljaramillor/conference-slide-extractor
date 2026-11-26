"""Detect temporary blended captures bracketed by independent clean content."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from pathlib import PurePosixPath
import re

import cv2
import numpy as np


EXCLUSION_REASON = 'blended slide transition (auto-excluded)'
_TIMESTAMP = re.compile(r'image_(\d\d)-(\d\d)-(\d\d)', re.IGNORECASE)
_SEQUENCE = re.compile(r'((?:(?:capture|frame|image|screenshot)[_ -]?)?)(\d+)', re.IGNORECASE)
_PERSPECTIVE = {'auto-perspective', 'projected-screen'}


def _clock(item):
    path = PurePosixPath(item['name'].replace('\\', '/'))
    if match := _TIMESTAMP.fullmatch(path.stem):
        hour, minute, second = map(int, match.groups())
        if hour < 24 and minute < 60 and second < 60:
            return (path.parent, 'clock'), hour * 3600 + minute * 60 + second, 10
    if match := _SEQUENCE.fullmatch(path.stem):
        prefix, index = match.groups()
        return (path.parent, prefix.lower()), int(index), 1
    return None


def _adjacent(left, right, left_clock, right_clock):
    return bool(left_clock and right_clock and left_clock[0] == right_clock[0]
                and 0 < right_clock[1] - left_clock[1] <= left_clock[2]
                and ('id' not in left or 'id' not in right or right['id'] == left['id'] + 1))


@dataclass(frozen=True)
class _Blend:
    weight: float
    residual: float
    closest: float
    agreement: float
    active: float
    spread: float


def _blend(before, middle, after, mask=None):
    # Average slightly to tolerate JPEG noise and thin letter edges at different subpixel positions.
    a, b, c = (cv2.resize(frame, (240, 135), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
               for frame in (before, middle, after))
    delta = c - a
    active = np.abs(delta) >= .10
    if mask is None:
        mask = np.ones(a.shape, dtype=bool)
    else:
        mask = cv2.resize(mask, (240, 135), interpolation=cv2.INTER_AREA) == 255
        active &= mask
    if active.mean() < .0018:
        return None
    evidence = delta[active]
    weight = float(np.dot((b - a)[active], evidence) / np.dot(evidence, evidence))
    if not .13 <= weight <= .87:
        return None
    residual = np.abs(b - (a + weight * delta))
    closest = min(float(np.abs(b - a)[mask].mean()), float(np.abs(b - c)[mask].mean()))
    agreement = float((residual[active] <= .22 * np.abs(evidence) + .012).mean())
    tiles = active.reshape(9, 15, 12, 20).mean(axis=(1, 3))
    return _Blend(weight, float(residual[mask].mean()), closest, agreement,
                  float(active.mean()), float((tiles >= .08).mean()))


def _distinct(before, after):
    """A content layer must disappear and another must appear at the endpoints."""
    a, c = (cv2.Canny(frame, 60, 140) > 0 for frame in (before, after))
    kernel = np.ones((3, 3), dtype=np.uint8)
    expanded_a, expanded_c = (cv2.dilate(edges.astype(np.uint8), kernel) > 0 for edges in (a, c))
    return (int((a & ~expanded_c).sum()) >= max(150, .085 * int(a.sum()))
            and int((c & ~expanded_a).sum()) >= max(150, .085 * int(c.sum())))


def _strong(fit):
    broad = fit.active >= .14 and fit.spread >= .34
    tight = fit.active >= .035 and fit.spread >= .12 and fit.residual <= .008 and fit.agreement >= .94
    return (fit.closest >= .008 and fit.residual <= min(.05, .38 * fit.closest)
            and fit.agreement >= .82 and (broad or tight))


def _same_crop(first, second):
    return bool(first.get('crop') and second.get('crop')
                and first.get('crop_mode') == second.get('crop_mode')
                and max(abs(a - b) for a, b in zip(first['crop'], second['crop'])) <= .045)


def _align_pane(detector, matcher, source, target):
    """Find a smaller pane in a neighboring frame expanded into the blended capture."""
    keys, descriptors = detector.detectAndCompute(source, None)
    target_keys, target_descriptors = detector.detectAndCompute(target, None)
    if descriptors is None or target_descriptors is None or len(target_descriptors) < 2:
        return None
    pairs = sorted((m for m, n in matcher.knnMatch(descriptors, target_descriptors, k=2)
                    if m.distance < .72 * n.distance), key=lambda match: match.distance)
    unique = {}
    for match in pairs:
        unique.setdefault(match.trainIdx, match)
    if len(unique) < 12:
        return None
    points = np.float32([keys[m.queryIdx].pt for m in unique.values()])
    targets = np.float32([target_keys[m.trainIdx].pt for m in unique.values()])
    matrix, inliers = cv2.estimateAffinePartial2D(points, targets, method=cv2.RANSAC, ransacReprojThreshold=4)
    if matrix is None:
        return None
    inliers = inliers.ravel().astype(bool)
    if inliers.sum() < max(12, .50 * len(unique)):
        return None
    height, width = target.shape
    span = np.ptp(targets[inliers], axis=0)
    if not (1.18 <= matrix[0, 0] <= 2.80 and abs(matrix[1, 0]) < .10
            and span[0] >= .38 * width and span[1] >= .26 * height):
        return None
    mask = cv2.warpAffine(np.full_like(source, 255), matrix, (width, height))
    if (mask == 255).mean() < .95:
        return None
    return cv2.warpAffine(source, matrix, (width, height)), mask


def _resized_blend(left, middle, right, detector, matcher):
    for same, other, reverse in ((left, right, False), (right, left, True)):
        if not _same_crop(same, middle) or _same_crop(other, middle):
            continue
        aligned = _align_pane(detector, matcher, other['gray'], middle['gray'])
        if aligned is None:
            continue
        registered, mask = aligned
        before, after = (registered, same['gray']) if reverse else (same['gray'], registered)
        fit = _blend(before, middle['gray'], after, mask)
        if fit and _strong(fit) and _distinct(before, after):
            return fit
    return None


def _record(items, first, center, last, kind, fit):
    middle = items[center]
    middle['crossfade'] = True
    middle['transition'] = {'kind': kind, 'before': items[first].get('id', first),
                            'after': items[last].get('id', last), 'weight': round(fit.weight, 4),
                            'residual': round(fit.residual, 4), 'agreement': round(fit.agreement, 4)}


def _resolve_dissolves(items):
    """Require a run of captures to fit the same clear endpoints in visual order."""
    positions = [index for index, item in enumerate(items) if item.get('crossfade')]
    for _, run in groupby(enumerate(positions), key=lambda entry: entry[1] - entry[0]):
        centers = [index for _, index in run]
        if len(centers) < 2:
            continue
        first, last = centers[0] - 1, centers[-1] + 1
        before, after = items[first]['gray'], items[last]['gray']
        if not _distinct(before, after):
            continue
        fits = [_blend(before, items[center]['gray'], after) for center in centers]
        if (any(fit is None or not _strong(fit) for fit in fits)
                or any(a.weight > b.weight + .015 for a, b in zip(fits, fits[1:]))):
            continue
        for center, fit in zip(centers, fits):
            _record(items, first, center, last, 'crossfade', fit)


def mark_crossfades(items):
    """Mark very clean partial reveals softly and omit proven two-slide blends."""
    for item in items:
        item.pop('crossfade', None)
        item.pop('transition', None)
    clocks = [_clock(item) for item in items]
    detector = matcher = None
    for index in range(1, len(items) - 1):
        left, middle, right = items[index - 1:index + 2]
        if (not _adjacent(left, middle, clocks[index - 1], clocks[index])
                or not _adjacent(middle, right, clocks[index], clocks[index + 1])
                or any(item.get('crop_mode') in _PERSPECTIVE for item in (left, middle, right))):
            continue
        before, captured, after = (item['gray'] for item in (left, middle, right))
        fit = _blend(before, captured, after)
        if fit and fit.residual <= .005 and fit.agreement >= .95:
            middle['crossfade'] = True
        kind = 'crossfade'
        if not (fit and _strong(fit) and _distinct(before, after)):
            if not ((_same_crop(left, middle) and not _same_crop(right, middle))
                    or (_same_crop(right, middle) and not _same_crop(left, middle))):
                continue
            if min(float(cv2.norm(captured, endpoint, cv2.NORM_L1)) / captured.size for endpoint in (before, after)) < 5:
                continue
            if detector is None:
                detector = cv2.SIFT_create(nfeatures=1600, contrastThreshold=.012)
                matcher = cv2.BFMatcher(cv2.NORM_L2)
            fit = _resized_blend(left, middle, right, detector, matcher)
            if fit is None:
                continue
            kind = 'resized-crossfade'
        _record(items, index - 1, index, index + 1, kind, fit)
    _resolve_dissolves(items)
    # Each automatic exclusion needs both endpoint captures available for export.
    inferred = {item.get('id', index) for index, item in enumerate(items) if item.get('transition')}
    for item in items:
        if evidence := item.get('transition'):
            if evidence['before'] in inferred or evidence['after'] in inferred:
                del item['transition']
    return [item for item in items if item.get('transition')]
