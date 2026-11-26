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


def _flat_color_fractions(image):
    colors = (image[::2, ::2].reshape(-1, 3) // 16).astype(np.int32)
    palette = np.bincount(colors @ [1, 16, 256], minlength=4096)
    return palette.max() / len(colors), np.partition(palette, -3)[-3:].sum() / len(colors)


def _overwrites_printed_content(left, right, delta, lost, added):
    """Detect a solid callout drawn over ink already present on the slide."""
    height, width = delta.shape
    closed = cv2.morphologyEx((delta > 24).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (not max(14, .075 * width) <= w <= .55 * width or h < max(14, .045 * height)
                or cv2.contourArea(contour) < .72 * w * h):
            continue
        region = np.s_[y+5:y+h-5, x+5:x+w-5]
        # Ignore the border. A callout must both erase ink and add new interior ink.
        if min(np.count_nonzero(lost[region]), np.count_nonzero(added[region])) < 150:
            continue
        flat = [_flat_color_fractions(image[region]) for image in (left, right)]
        if (min(total for _, total in flat) >= .34
                and any(dominant >= .29 and total >= .50 for dominant, total in flat)):
            return True
    return False


def has_local_content_replacement(left, right, *, check_overlays=False, overlays_only=False):
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
    if check_overlays and _overwrites_printed_content(left, right, delta, lost, added):
        return True
    if overlays_only:
        return False
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
        # Fade-ins often change brightness in one direction. A countdown can
        # also erase far more ink than it adds.
        if min(lighter, darker) < max(85, max(lighter, darker) * .25):
            continue
        # A slide region usually has a small set of flat colors. Do not
        # treat presenter movement or video within a slide as replaced text.
        if all(_flat_color_fractions(image[region])[1] >= .67 for image in (left, right)):
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


def _projected_alignment(clean_gray, projected_gray):
    """Find a frame-wide alignment between a slide and its on-stage projection."""
    height, width = clean_gray.shape
    equalizer = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8))
    detector = cv2.SIFT_create(nfeatures=1800, contrastThreshold=.014)
    keypoints, descriptors = zip(*(detector.detectAndCompute(equalizer.apply(gray), None)
                                  for gray in (clean_gray, projected_gray)))
    if any(vector is None or len(vector) < 80 for vector in descriptors):
        return None
    matcher = cv2.BFMatcher()
    forward, backward = (
        {best.queryIdx: best.trainIdx for best, alternate in matcher.knnMatch(first, second, k=2)
         if best.distance < .74 * alternate.distance}
        for first, second in (descriptors, descriptors[::-1])
    )
    matches = [(a, b) for a, b in forward.items() if backward.get(b) == a]
    if len(matches) < 80:
        return None
    left, right = (np.float32([points[match[side]].pt for match in matches])
                   for side, points in enumerate(keypoints))
    transform, inliers = cv2.findHomography(right, left, cv2.RANSAC, 1.8)
    if transform is None or inliers is None or inliers.mean() < .80:
        return None
    reliable = left[inliers.ravel() > 0]
    quadrant = (reliable > [width / 2, height / 2]) @ [1, 2]
    if ((np.bincount(quadrant, minlength=4) < 12).any()
            or (np.ptp(reliable, axis=0) < [.75 * width, .72 * height]).any()):
        return None
    corners = np.float32([[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]])
    distance = np.abs(cv2.perspectiveTransform(corners, transform) - corners)
    if (distance > [width * .025, height * .025]).any():
        return None
    return transform


def _same_projected_print(clean_gray, projected_gray, valid):
    """Require matching local ink despite projector blur and minor lens distortion."""
    def detail(gray, sigma):
        gray = gray.astype(np.float32)
        return cv2.GaussianBlur(gray, (0, 0), sigma) - cv2.GaussianBlur(gray, (0, 0), 3.5)

    def average(plane, size=7):
        return cv2.boxFilter(plane, -1, (size, size))

    clean = detail(clean_gray, 1.15)
    projected = detail(projected_gray, .8)
    for first, second in ((clean, projected), (projected, clean)):
        activity = average(first * first)
        best = np.full_like(first, -1)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                shifted = np.roll(second, (dy, dx), axis=(0, 1))
                support = np.sqrt(np.maximum(.0001, activity * average(shifted * shifted)))
                correlation = average(first * shifted) / support
                np.maximum(best, correlation, out=best)
        unmatched = valid & (((best < .35) & (activity > 12)) | ((best < .50) & (activity > 40)))
        _, _, regions, _ = cv2.connectedComponentsWithStats(unmatched.astype(np.uint8), connectivity=8)
        if any(w >= 2 and h >= 2 and area >= 9 for _, _, w, h, area in regions[1:]):
            return False

    # Neighboring unchanged digits can dominate the tolerant comparison.
    # Check for a tight residual above the noise measured across this slide.
    clean, projected = (detail(gray, .85) for gray in (clean_gray, projected_gray))
    energy_a, energy_b, shared = (average(plane, 5) for plane in
                                  (clean * clean, projected * projected, clean * projected))
    power = np.maximum(energy_a, energy_b)
    active = valid & (power > 16)
    if not active.any():
        return False
    similarity = shared / np.sqrt(np.maximum(.0001, energy_a * energy_b))
    residual = (1 - similarity) * np.sqrt(power)
    threshold = max(2.5, 3.1 * np.quantile(residual[active], .99))
    isolated = active & (residual > threshold)
    _, _, regions, _ = cv2.connectedComponentsWithStats(isolated.astype(np.uint8), connectivity=8)
    return not any(w >= 2 and h >= 2 and area >= 5 for _, _, w, h, area in regions[1:])


def _same_projected_colors(clean, projected, valid):
    """Check that the residual colors fit a whole-frame tint and light falloff."""
    height, width = valid.shape
    clean, projected = (cv2.GaussianBlur(image, (7, 7), 0).astype(np.float32) / 255
                        for image in (clean, projected))
    if np.mean(np.abs(clean - projected)[valid]) < .04:
        return False
    y, x = np.meshgrid(np.linspace(-1, 1, height, dtype=np.float32),
                       np.linspace(-1, 1, width, dtype=np.float32), indexing='ij')
    blue, green, red = clean.transpose(2, 0, 1)
    luminance = .114 * blue + .587 * green + .299 * red
    spatial = (x, y, x * y, x * x, y * y)
    color = (blue, green, red, blue * blue, green * green, red * red,
             blue * green, blue * red, green * red)
    features = np.stack((np.ones_like(blue), *color, *spatial,
                         *(luminance * field for field in spatial)), axis=2)
    sampled = features[valid][::13]
    observed = projected[valid][::13]
    for _ in range(3):
        coefficients = np.linalg.lstsq(sampled, observed, rcond=.0005)[0]
        error = np.max(np.abs(sampled @ coefficients - observed), axis=1)
        regular = error <= np.quantile(error, .965)
        sampled, observed = sampled[regular], observed[regular]
    residual = np.max(np.abs(features @ coefficients - projected), axis=2)
    return bool((np.quantile(residual[valid], [.90, .99, .999]) <= [15 / 255, 34 / 255, 52 / 255]).all())


def tint_duplicate(left, right):
    """Verify a nearby projected capture repeats the same printed slide."""
    clean, projected = sorted((left, right), key=lambda image: image.shape[0] * image.shape[1], reverse=True)
    height, width = projected.shape[:2]
    if (height < 300 or min(width, clean.shape[1]) < 550
            or abs((clean.shape[1] / clean.shape[0]) / (width / height) - 1) > .08
            or clean.shape[0] * clean.shape[1] < 1.35 * height * width):
        return False
    scale = min(1., 780 / width)
    width, height = round(width * scale), round(height * scale)
    size = (width, height)
    clean, projected = (cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                        for image in (clean, projected))
    clean_gray, projected_gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                  for image in (clean, projected))
    transform = _projected_alignment(clean_gray, projected_gray)
    if transform is None:
        return False
    projected = cv2.warpPerspective(projected, transform, size)
    projected_gray = cv2.cvtColor(projected, cv2.COLOR_BGR2GRAY)
    valid = cv2.warpPerspective(np.full((height, width), 255, np.uint8), transform, size) == 255
    if valid.mean() < .96:
        return False
    valid = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    # Exclude the projected crop's edge and the boundary of the warp.
    valid[:12, :] = valid[-12:, :] = valid[:, :12] = valid[:, -12:] = False
    return (_same_projected_print(clean_gray, projected_gray, valid)
            and _same_projected_colors(clean, projected, valid))


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
