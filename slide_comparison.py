"""Register preview crops and map visible changes back to each native image."""

import cv2
import numpy as np

from slide_ocr import text_evidence


def _result(status, relationship='uncertain', left=None, right=None, numeric=False):
    return {
        'status': status,
        'relationship': relationship,
        'regions': {'left': left or [], 'right': right or []},
        'numeric_conflict': numeric,
    }


def _edges(image):
    return cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 38, 100) > 0


def _edge_overlap(left, right, valid):
    kernel = np.ones((3, 3), np.uint8)
    expanded_left = cv2.dilate(left.astype(np.uint8), kernel) > 0
    expanded_right = cv2.dilate(right.astype(np.uint8), kernel) > 0
    left_shared = np.count_nonzero(left & expanded_right & valid)
    right_shared = np.count_nonzero(right & expanded_left & valid)
    count = min(left_shared, right_shared)
    coverage = max(
        left_shared / max(1, np.count_nonzero(left & valid)),
        right_shared / max(1, np.count_nonzero(right & valid)),
    )
    return count, coverage


def _register(left, right, left_edges, right_edges):
    """Allow a small translation when the crops contain convincing common features."""
    height, width = left.shape[:2]
    valid = np.ones((height, width), np.uint8)
    direct_count, direct_cover = _edge_overlap(left_edges, right_edges, valid)
    if direct_cover >= .96:
        return right, right_edges, valid.astype(bool), (0, 0), direct_count, direct_cover

    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    offset, confidence = cv2.phaseCorrelate(left_edges.astype(np.float32),
                                            right_edges.astype(np.float32), window)
    if confidence >= .18 and np.isfinite(offset).all():
        x, y = (round(value) for value in offset)
        if (x or y) and abs(x) <= width * .025 and abs(y) <= height * .025:
            matrix = np.float32([[1, 0, -x], [0, 1, -y]])
            aligned_edges = cv2.warpAffine(right_edges.astype(np.uint8), matrix, (width, height)) > 0
            aligned_valid = cv2.warpAffine(valid, matrix, (width, height)) > 0
            count, coverage = _edge_overlap(left_edges, aligned_edges, aligned_valid)
            if coverage >= direct_cover + .06 and count >= max(45, direct_count):
                aligned = cv2.warpAffine(right, matrix, (width, height))
                return aligned, aligned_edges, aligned_valid, (x, y), count, coverage
    return right, right_edges, valid.astype(bool), (0, 0), direct_count, direct_cover


def _change_regions(raw_changes):
    expanded = cv2.dilate(raw_changes.astype(np.uint8), np.ones((19, 15), np.uint8))
    contours, _ = cv2.findContours(expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    retained_changes = np.zeros_like(raw_changes)
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        active = raw_changes[y:y + height, x:x + width]
        if np.count_nonzero(active) >= 7:
            regions.append((x, y, x + width, y + height))
            retained_changes[y:y + height, x:x + width] |= active
    return sorted(regions, key=lambda box: (box[1], box[0])), retained_changes


def _normalized_regions(regions, shape, offset=(0, 0)):
    height, width = shape
    x, y = offset
    result = []
    for x1, y1, x2, y2 in regions:
        coordinates = ((x1 + x) / width, (y1 + y) / height,
                       (x2 + x) / width, (y2 + y) / height)
        result.append([round(min(1., max(0., value)), 5) for value in coordinates])
    return result


def _prior_content_untouched(image, edges, changed, valid):
    """Require changes to sit clear of the earlier stage's existing content."""
    background = np.median(image[valid][::5], axis=0).astype(np.uint8)
    foreground = np.max(cv2.absdiff(image, np.broadcast_to(background, image.shape).copy()), axis=2) > 30
    # A new digit or a changed panel close to earlier ink is not evidence of a
    # complete build. The wider horizontal margin covers adjacent digits.
    neighborhood = cv2.dilate(changed.astype(np.uint8), np.ones((7, 21), np.uint8)) > 0
    return np.count_nonzero((foreground | edges) & neighborhood & valid) <= 3


def _has_replacement(left_edges, right_edges, changed):
    nearby = cv2.dilate(changed.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    kernel = np.ones((3, 3), np.uint8)
    for old, new in ((left_edges, right_edges), (right_edges, left_edges)):
        matched = cv2.dilate(new.astype(np.uint8), kernel) > 0
        if np.count_nonzero(old & ~matched & nearby) < 12:
            return False
    return True


def has_local_content_replacement(left, right):
    """Find a replaced text or chart region inside otherwise matching captures."""
    if left.shape != right.shape:
        return False
    height, width = left.shape[:2]
    scale = min(1., 960 / max(height, width))
    size = (round(width * scale), round(height * scale))
    left, right = (cv2.GaussianBlur(cv2.resize(image, size, interpolation=cv2.INTER_AREA), (3, 3), 0)
                   for image in (left, right))
    delta = cv2.absdiff(left, right).max(axis=2)
    regions, changed = _change_regions(delta > 24)
    if not regions:
        return False

    left_edges, right_edges = map(_edges, (left, right))
    expanded_change = cv2.dilate(changed.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    kernel = np.ones((3, 3), np.uint8)
    lost, added = (old & ~(cv2.dilate(new.astype(np.uint8), kernel) > 0) & expanded_change
                   for old, new in ((left_edges, right_edges), (right_edges, left_edges)))
    gray_left, gray_right = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in (left, right))
    for x1, y1, x2, y2 in regions:
        region = np.s_[y1:y2, x1:x2]
        if x2 - x1 < size[0] * .08 or np.count_nonzero(delta[region] > 42) < 140:
            continue
        removed_count = np.count_nonzero(lost[region])
        new_count = np.count_nonzero(added[region])
        # Ignore small pointer marks, JPEG fringes, and elements becoming sharper
        # while keeping their shape. Require loss of a meaningful fraction of ink.
        if (removed_count < max(75, .08 * np.count_nonzero(left_edges[region]))
                or new_count < max(75, .08 * np.count_nonzero(right_edges[region]))):
            continue
        old, new = gray_left[region], gray_right[region]
        signed = old.astype(np.int16) - new.astype(np.int16)
        lighter, darker = np.count_nonzero(signed > 25), np.count_nonzero(signed < -25)
        # Fade-ins and opaque overlays produce changes almost entirely in
        # one direction. A countdown can also erase far more ink than it adds.
        if min(lighter, darker) < max(85, max(lighter, darker) * .25):
            continue
        # A slide region usually has a small set of flat colors. Do not
        # treat presenter movement or video within a slide as replaced text.
        for image in (left, right):
            colors = (image[region][::2, ::2].reshape(-1, 3) // 16).astype(np.int32)
            palette = np.bincount(colors @ [1, 16, 256], minlength=4096)
            if np.partition(palette, -3)[-3:].sum() < .67 * len(colors):
                break
        else:
            # A small geometric shift can occur when a new footer or QR code moves
            # surrounding text. Check the affected region for matching structure.
            window = cv2.createHanningWindow((x2 - x1, y2 - y1), cv2.CV_32F)
            offset, confidence = cv2.phaseCorrelate(old.astype(np.float32), new.astype(np.float32), window)
            if confidence < .78 or not 1.3 <= max(map(abs, offset)) <= .015 * size[0]:
                return True
    return False


def _preserves_native_content(earlier, later):
    """Veto a recommendation when native pixels show changes to earlier content."""
    if earlier.shape != later.shape:
        return False
    old, new = (cv2.GaussianBlur(image, (3, 3), 0) for image in (earlier, later))
    active = cv2.absdiff(old, new).max(axis=2) > 30
    _, meaningful = _change_regions(active)
    return _prior_content_untouched(old, _edges(old), meaningful, np.ones(active.shape, dtype=bool))


def has_added_chart_outlines(earlier, later):
    """Detect thin colored chart outlines on two equally sized native crops."""
    if earlier.shape != later.shape or min(earlier.shape[:2]) < 270:
        return False
    height, width = earlier.shape[:2]
    earlier, later = (cv2.GaussianBlur(image, (3, 3), 0) for image in (earlier, later))
    changed = cv2.absdiff(earlier, later).max(axis=2) > 27
    count = np.count_nonzero(changed)
    if not .003 <= count / changed.size <= .018:
        return False

    # Bridge interruptions at chart intersections, then retain long strokes
    # along each axis. A solid panel has substantial support in both axes.
    pixels = changed.astype(np.uint8)
    gap = max(3, 2 * round(width * .003) + 1)
    horizontal_length = 2 * round(width * .0225) + 1
    vertical_length = 2 * round(height * .0375) + 1
    horizontal = cv2.morphologyEx(pixels, cv2.MORPH_CLOSE, np.ones((1, gap), np.uint8))
    vertical = cv2.morphologyEx(pixels, cv2.MORPH_CLOSE, np.ones((gap, 1), np.uint8))
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, np.ones((1, horizontal_length), np.uint8)) > 0
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, np.ones((vertical_length, 1), np.uint8)) > 0
    if (min(np.count_nonzero(horizontal & changed), np.count_nonzero(vertical & changed)) < count * .10
            or np.count_nonzero(horizontal & vertical & changed) > count * .10):
        return False

    strokes = cv2.dilate((horizontal | vertical).astype(np.uint8), np.ones((gap, gap), np.uint8)) > 0
    supported = changed & strokes
    supported_count = np.count_nonzero(supported)
    # A small pointer movement may accompany a reveal. Require the remaining
    # changes to follow the detected outlines.
    if count - supported_count > min(count * .025, changed.size * .00010):
        return False

    old_saturation, new_saturation = (cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.int16)
                                      for image in (earlier, later))
    increased = new_saturation - old_saturation > 16
    return np.count_nonzero(increased & supported) >= supported_count * .86


def compare_captures(left, right, left_item=None, right_item=None):
    """Return visible regions and conservative, advisory build evidence."""
    left_height, left_width = left.shape[:2]
    right_height, right_width = right.shape[:2]
    if min(left_height, left_width, right_height, right_width) < 75:
        return _result('uncertain')
    aspect = left_width / left_height
    if abs((right_width / right_height) / aspect - 1) > .035:
        return _result('uncertain')
    if np.array_equal(left, right):
        return _result('ok', 'same-content')
    native_left, native_right = left, right
    long_edge = min(960, max(left_height, left_width), max(right_height, right_width))
    scale = long_edge / max(left_height, left_width)
    size = (round(left_width * scale), round(left_height * scale))

    # Damp the resampling fringes when two presentation layouts crop to
    # different native sizes. The full-resolution previews are untouched.
    kernel = (3, 3) if left.shape == right.shape else (5, 5)
    left, right = (cv2.GaussianBlur(cv2.resize(image, size, interpolation=cv2.INTER_AREA), kernel, 0)
                   for image in (left, right))
    left_edges, right_edges = map(_edges, (left, right))
    right, right_edges, valid, offset, shared, coverage = _register(left, right, left_edges, right_edges)
    raw_changes = (cv2.absdiff(left, right).max(axis=2) > 24) & valid
    regions, changed = _change_regions(raw_changes)
    if shared < 45 or coverage < .64 or len(regions) > 24:
        return _result('uncertain')

    evidence = {}
    if left_item and right_item and all(item.get('ocr', {}).get('status') == 'ok'
                                        for item in (left_item, right_item)):
        evidence = text_evidence(left_item.copy(), right_item.copy(), native_pair=lambda: (native_left, native_right))
    numeric = bool(evidence.get('numeric_conflict'))
    left_boxes = _normalized_regions(regions, changed.shape)
    right_boxes = _normalized_regions(regions, changed.shape, offset)
    if numeric:
        return _result('ok', 'content-replaced', left_boxes, right_boxes, numeric=True)

    perspective = any(item and item.get('crop_mode') in {'projected-screen', 'auto-perspective'}
                      for item in (left_item, right_item))
    if offset == (0, 0) and not perspective and np.count_nonzero(changed) >= 20:
        right_added = _prior_content_untouched(left, left_edges, changed, valid)
        left_added = _prior_content_untouched(right, right_edges, changed, valid)
        if (right_added and not left_added and evidence.get('text_cover_a', 1) >= .97
                and _preserves_native_content(native_left, native_right)):
            return _result('ok', 'right-complete', left_boxes, right_boxes)
        if (left_added and not right_added and evidence.get('text_cover_b', 1) >= .97
                and _preserves_native_content(native_right, native_left)):
            return _result('ok', 'left-complete', left_boxes, right_boxes)
    relationship = 'content-replaced' if _has_replacement(left_edges, right_edges, changed) else 'uncertain'
    return _result('ok', relationship, left_boxes, right_boxes)
