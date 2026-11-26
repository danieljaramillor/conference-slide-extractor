"""Local conference slide deduplication. No network or API calls."""
from __future__ import annotations

import argparse
from collections import defaultdict
import concurrent.futures
import csv
from functools import lru_cache
import hashlib
import io
import json
import os
import platform
from pathlib import Path
import re
import time
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageOps

from slide_comparison import has_added_chart_outlines, has_local_content_replacement, tint_duplicate
from slide_pages import confirm_footer_numbers, footer_number
from slide_regions import broadcast_crop_profile, confirmed_presenter_crops, detect_region, extract, presenter_layout, quality, recover_projected_screen
from slide_text import SESSION_TEXT, slide_text_record
from slide_transitions import EXCLUSION_REASON, capture_runs, mark_crossfades, mark_one_sided_transitions

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
APP_VERSION = '2.15.0'
cv2.setNumThreads(1)


def natural_key(value):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r'(\d+)', str(value))]


class Source:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.zip = zipfile.ZipFile(self.path) if self.path.is_file() else None
        if self.zip:
            names = [i.filename for i in self.zip.infolist() if not i.is_dir()]
            if len(names) != len(set(names)):
                raise ValueError('ZIP contains duplicate filenames; rename them before processing.')
        else:
            names = [p.relative_to(self.path).as_posix() for p in self.path.rglob('*') if p.is_file()]
        self.names = sorted([n for n in names if Path(n).suffix.lower() in EXTENSIONS], key=natural_key)

    def read(self, name):
        return self.zip.read(name) if self.zip else (self.path / name).read_bytes()

    def close(self):
        if self.zip:
            self.zip.close()


def decode(data):
    with Image.open(io.BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def detect_crop(image):
    """Compatibility helper; use detect_region for perspective/quality metadata."""
    region = detect_region(image)
    return region['crop'], region['crop_mode']


def crop_image(image, box):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    return image[round(y1*h):round(y2*h), round(x1*w):round(x2*w)]


def features(image):
    normalized = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 60, 140)
    # Title helps distinguish content builds from unrelated slides in the same template.
    descriptor = cv2.resize(normalized, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32).flatten() / 255
    return normalized, gray, edges, descriptor


def compare(a, b, read_native=None):
    delta = np.abs(a['gray'].astype(np.int16) - b['gray'].astype(np.int16))
    changed = float(np.mean(delta > 28))
    mae = float(delta.mean() / 255)
    ea, eb = a['edges'] > 0, b['edges'] > 0
    da = cv2.dilate(a['edges'], np.ones((3, 3), np.uint8)) > 0
    db = cv2.dilate(b['edges'], np.ones((3, 3), np.uint8)) > 0
    cover_a = float(np.sum(ea & db) / max(1, ea.sum()))
    cover_b = float(np.sum(eb & da) / max(1, eb.sum()))
    title = float(np.mean(delta[:44] > 28))
    heading = float(np.mean(delta[:44, :320] > 28))
    # A local change can be a tiny numeric result, even with low global error.
    local_change = float((delta[:270,:480]>28).reshape(18,15,24,20).mean(axis=(1,3)).max())
    duplicate = (mae < .018 and changed < .035 and title < .025 and min(cover_a, cover_b) > .92)
    # A build must preserve the title and almost all of the earlier stage's edges.
    body_a, body_b = int(ea[54:245].sum()), int(eb[54:245].sum())
    light_header = min(float(np.mean(a['gray'][:44] > 190)), float(np.mean(b['gray'][:44] > 190))) > .65
    title_edges = min(int(ea[:44, :420].sum()), int(eb[:44, :420].sum())) > 180
    # The right edge may acquire a QR code or new sidebar during a build.
    # Require stability in the heading itself and containment across the whole slide.
    build = (light_header and title_edges and heading < .008 and max(cover_a, cover_b) > .965
             and min(cover_a, cover_b) > .40 and changed < .36
             and min(body_a, body_b) > 200)
    replacement = False
    if changed > .004 and min(cover_a, cover_b) > .80:
        # In a repeated template, a dense block may disappear as a different
        # dense block appears. Global similarity hides this on sparse tables.
        removed = (ea & ~db).reshape(9, 30, 12, 40).sum(axis=(1, 3))[2:8]
        added = (eb & ~da).reshape(9, 30, 12, 40).sum(axis=(1, 3))[2:8]
        removed_total, added_total = int(removed.sum()), int(added.sum())
        replacement = bool(min(removed.max(), added.max()) >= 75
                           and min(removed_total, added_total) >= .55 * max(removed_total, added_total))
    # A small reveal can satisfy both old thresholds. If one frame retains
    # effectively all the other's edges and adds a localized block, treat it
    # as content before comparing size/sharpness of equivalent captures.
    localized_reveal = bool(duplicate and build and not replacement and local_change >= .20
                            and max(cover_a, cover_b) >= .98 and abs(cover_a - cover_b) >= .025)
    annotation = {}
    if (read_native is not None and duplicate and build and not replacement and not localized_reveal
            and .10 <= local_change and max(cover_a, cover_b) >= .985 and abs(cover_a - cover_b) >= .012
            and a.get('_presenter_pane') and b.get('_presenter_pane') and a['crop'] == b['crop']):
        first_has_more = cover_b > cover_a
        earlier, later = (b, a) if first_has_more else (a, b)
        if has_added_chart_outlines(read_native(earlier), read_native(later)):
            annotation['prefer_first_for_outline'] = first_has_more
            localized_reveal = True
    trusted_crop = min(a.get('crop_confidence', 0), b.get('crop_confidence', 0)) >= .85
    small_replacement = (changed < .10 and local_change >= .09
                         and max(cover_a, cover_b) >= .94 and min(cover_a, cover_b) >= .80)
    # An opaque callout may overwrite a label while a much larger panel appears
    # beside it. The panel can dominate global counts even in a verified crop.
    wide_build = (build and trusted_crop and changed < .20 and local_change >= .60
                  and max(cover_a, cover_b) >= .975 and not small_replacement)
    if (read_native is not None and not replacement and not annotation and (small_replacement or wide_build)
            and a.get('crop') == b.get('crop') and a.get('quad') is None and b.get('quad') is None
            and a.get('slide_width') == b.get('slide_width') and a.get('slide_height') == b.get('slide_height')):
        check_overlays = wide_build or (.008 <= changed and local_change >= .30 and trusted_crop)
        replacement = has_local_content_replacement(read_native(a), read_native(b),
                                                    check_overlays=check_overlays, overlays_only=wide_build)
    return {'mae': round(mae, 5), 'changed': round(changed, 5),
            'coverage_a': round(cover_a, 4), 'coverage_b': round(cover_b, 4),
            'title_changed': round(title, 5), 'local_change':round(local_change,4),
            'duplicate': duplicate and not replacement and not localized_reveal,
            'build': bool(build and not replacement),
            'content_replacement': replacement, 'localized_reveal': localized_reveal and not replacement, **annotation}


def slide_features(slide):
    normalized, gray, edges, descriptor = features(slide)
    return {**quality(slide), 'normalized': normalized, 'gray': gray, 'edges': edges, 'descriptor': descriptor,
            'detail': int((edges[54:245] > 0).sum())}


def prepare(source, index, name, crop):
    data=source.read(name)
    image = decode(data)
    region = {'crop':crop,'quad':None,'crop_mode':'manual','crop_confidence':1.,
              'region_ambiguous':False,'region_candidates':1} if crop else detect_region(image)
    slide = extract(image, region)
    if not crop:
        recovered = recover_projected_screen(slide, region)
        if recovered is not None:
            region = recovered
            slide = extract(image, region)
    captures_most_of_frame = slide.shape[0] * slide.shape[1] >= .90 * image.shape[0] * image.shape[1]
    existing_inset_is_reliable = (region['crop_mode'] == 'auto-inset' and region['crop_confidence'] >= .85
                                  and not region['region_ambiguous'] and not captures_most_of_frame)
    layout = presenter_layout(image) if not crop and region['quad'] is None and (
        captures_most_of_frame or existing_inset_is_reliable) else None
    known_crop = region['crop'] if existing_inset_is_reliable else None
    presenter_pane = known_crop is not None and layout is not None and max(
        abs(a - b) for a, b in zip(known_crop, layout.crop)) <= .003
    return {'id': index, 'name': name, 'sha256': hashlib.sha256(data).hexdigest(), **region, **slide_features(slide),
            'width': image.shape[1], 'height': image.shape[0], '_presenter_layout': layout,
            '_presenter_known_crop': known_crop, '_presenter_pane': presenter_pane}


def apply_presenter_crops(source, items, pool, thumbs):
    anchors = defaultdict(list)
    for item in items:
        if item['_presenter_known_crop'] is not None and item['crop_confidence'] >= .90:
            anchors[item['_presenter_known_crop']].append(item)
    crops = []
    for captures in capture_runs(items):
        layouts = [item.pop('_presenter_layout') for item in captures]
        known_crops = [item.pop('_presenter_known_crop') for item in captures]
        crops.extend((captures[index], crop) for index, crop in confirmed_presenter_crops(layouts, known_crops).items())

    # A slide can defeat single-frame detection while still sitting inside the
    # same verified broadcast layout, particularly during a camera cut.
    resolved = {item['id'] for item, _ in crops}
    unresolved = [item for item in items if item['id'] not in resolved and item.get('quad') is None
                  and item['slide_pixels'] >= .90*item['width']*item['height']]
    profiles = []
    for crop, frames in anchors.items():
        if not unresolved or len(frames) < 3:
            continue
        indices = np.linspace(0, len(frames)-1, min(8, len(frames)), dtype=int)
        sampled = [profile for index in indices if (profile := broadcast_crop_profile(
            decode(source.read(frames[index]['name'])), crop)) is not None]
        for profile in sampled:
            if sum(np.mean(cv2.absdiff(profile, other)) <= 4 for other in sampled) >= 3:
                profiles.append((crop, profile))
                break

    def verify_layout(item):
        image = decode(source.read(item['name']))
        matches = [crop for crop, anchor in profiles if (observed := broadcast_crop_profile(image, crop, require_canvas=True)) is not None
                   and np.mean(cv2.absdiff(observed, anchor)) <= 4]
        return (item, matches[0]) if len(matches) == 1 else None

    if profiles:
        crops.extend(task for task in pool.map(verify_layout, unresolved) if task is not None)

    def prepare_crop(task):
        item, crop = task
        data = source.read(item['name'])
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError(f"Source image changed while scanning: {item['name']}")
        region = {'crop': crop, 'quad': None, 'crop_mode': 'presenter-slide',
                  'crop_confidence': .94, 'region_ambiguous': False, '_presenter_pane': True}
        return item, {**region, **slide_features(extract(decode(data), region))}

    for item, cropped in pool.map(prepare_crop, crops):
        write_image(thumbs / f"{item['id']}.jpg", cropped.pop('normalized'))
        item.update(cropped)
    return len(crops)


def better_frame(item, ref, metrics):
    if bool(item.get('crossfade')) != bool(ref.get('crossfade')):
        return not item.get('crossfade')
    if metrics.get('color_cast_verified'):
        return item['quad'] is None
    if 'prefer_first_for_outline' in metrics:
        return metrics['prefer_first_for_outline']
    if metrics.get('build') and not metrics.get('duplicate'):
        # Choose the superset, not a noisy earlier frame with higher sharpness.
        if metrics['coverage_b'] > metrics['coverage_a']+.025:
            return True
        if metrics['coverage_a'] > metrics['coverage_b']+.025:
            return False
        return item['detail'] > ref['detail']*1.04
    # Compare quality only after establishing equivalent slide content.
    iq=item['slide_pixels']*min(1.,max(.1,item['sharpness']/120))
    rq=ref['slide_pixels']*min(1.,max(.1,ref['sharpness']/120))
    return iq > rq


def match_projected_slide(item, neighbors, by_id, read_native):
    checked = set()
    for group in neighbors:
        if group is None or group['id'] in checked:
            continue
        checked.add(group['id'])
        ref = by_id[group['representative']]
        photo, direct = (item, ref) if item.get('quad') is not None else (ref, item)
        if (photo.get('crop_mode') != 'projected-screen' or photo.get('quad') is None
                or direct.get('quad') is not None or direct['crop_confidence'] < .85
                or direct['region_ambiguous'] or direct['slide_pixels'] < 1.35 * photo['slide_pixels']):
            continue
        photo_aspect = photo['slide_width'] / photo['slide_height']
        direct_aspect = direct['slide_width'] / direct['slide_height']
        if abs(photo_aspect / direct_aspect - 1) > .08:
            continue
        if tint_duplicate(read_native(photo), read_native(direct)):
            metrics = compare(item, ref)
            metrics.update(duplicate=True, build=False, content_replacement=False,
                           localized_reveal=False, color_cast_verified=True)
            return metrics['mae'], group, metrics
    return None


def write_image(path, image):
    ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError(f'Cannot encode {path}')
    path.write_bytes(data.tobytes())


def run(args):
    start = time.perf_counter()
    source = Source(args.input)
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output directory is not empty: {out}. Use a new output directory.')
    if not source.zip and out.is_relative_to(source.path):
        raise ValueError('Output must be outside the input image directory.')
    out.mkdir(parents=True, exist_ok=True)
    thumbs = out / 'thumbs'
    thumbs.mkdir(exist_ok=True)
    items, errors = [], []
    print(f'Processing {len(source.names)} images with {args.workers} workers...', flush=True)
    def work(pair):
        i, name = pair
        try:
            return prepare(source, i, name, args.crop)
        except Exception as exc:
            return {'error': str(exc), 'id': i, 'name': name}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for item in pool.map(work, enumerate(source.names)):
            if 'error' in item:
                errors.append(item)
            else:
                write_image(thumbs / f"{item['id']}.jpg", item.pop('normalized'))
                items.append(item)
            if (len(items) + len(errors)) % 200 == 0:
                print(f'  Read {len(items) + len(errors)}/{len(source.names)}', flush=True)
        presenter_crops = apply_presenter_crops(source, items, pool, thumbs) if args.crop is None else 0
    read_seconds = time.perf_counter() - start
    if not items:
        raise ValueError(f'No readable images. Errors: {errors[:3]}')
    items_by_id = {item['id']: item for item in items}

    @lru_cache(maxsize=12)
    def cached_native(image_id):
        capture = items_by_id[image_id]
        return extract(decode(source.read(capture['name'])), capture)

    def read_native(item):
        return cached_native(item['id'])
    mark_crossfades(items)
    mark_one_sided_transitions(items, read_capture=read_native)
    transitions = [item for item in items if item.get('transition')]
    transition_ids = {item['id'] for item in transitions}
    eligible = [item for item in items if item['id'] not in transition_ids]
    groups, matchable, decisions, candidates = [], [], [], []
    recent_groups = {}
    by_id = {i['id']: i for i in eligible}
    for pos, item in enumerate(eligible):
        item['position'] = pos
    descriptors = np.stack([i['descriptor'] for i in eligible])
    # Global lookup handles repeated slides even when filenames are not chronological.
    for pos, item in enumerate(eligible):
        # This blend can be the only capture of the outgoing content.
        protected = bool(item.get('transition_review'))
        if protected or not matchable:
            group = {'id': len(groups)+1, 'representative': item['id'], 'members': [item['id']], 'builds': 0}
            groups.append(group)
            if not protected:
                matchable.append(group)
                recent_groups[item['id']] = group
            continue
        rep_ids = [g['representative'] for g in matchable]
        rep_pos = [by_id[i]['position'] for i in rep_ids]
        distances = np.abs(descriptors[rep_pos] - item['descriptor']).mean(axis=1)
        best = None
        for gi in np.argsort(distances)[:min(8, len(matchable))]:
            if distances[gi] > .16:
                continue
            group = matchable[int(gi)]
            ref = by_id[group['representative']]
            metrics = compare(item, ref, read_native)
            match = metrics['duplicate'] or (args.builds == 'complete' and metrics['build'])
            if match:
                score = metrics['mae']
                if best is None or score < best[0]:
                    best = (score, group, metrics)
            elif metrics['mae'] < .055 or (metrics['title_changed'] < .008 and max(metrics['coverage_a'], metrics['coverage_b']) > .80):
                candidates.append({'a': ref['id'], 'b': item['id'], **metrics})
        if best is None:
            neighbors = (recent_groups.get(item['id'] - gap) for gap in range(1, 4))
            best = match_projected_slide(item, neighbors, by_id, read_native)
        if best:
            _, group, metrics = best
            ref = by_id[group['representative']]
            group['members'].append(item['id'])
            is_build = not metrics['duplicate']
            if is_build:
                group['builds'] += 1
            # Most content first, then sharpness for effectively identical frames.
            if better_frame(item, ref, metrics):
                group['representative'] = item['id']
            decisions.append({'image': item['id'], 'matched': ref['id'], 'group': group['id'],
                              'reason': 'build' if is_build else 'duplicate', **metrics})
        else:
            group = {'id': len(groups)+1, 'representative': item['id'], 'members': [item['id']], 'builds': 0}
            groups.append(group)
            matchable.append(group)
        recent_groups[item['id']] = group
    cached_native.cache_clear()
    from slide_audit import audit
    groups, candidates, audit_log = audit(source,eligible,groups,decisions,candidates,args,out)
    audit_log['presenter_crop'] = {'captures': presenter_crops}
    # Keep evidence and each original frame inspectable. Blends did not enter
    # matching or OCR, so they could not bridge two distinct slides.
    transition_exclusions = []
    next_group = max(group['id'] for group in groups) + 1
    for group_id, item in enumerate(transitions, next_group):
        groups.append({'id': group_id, 'representative': item['id'], 'members': [item['id']],
                       'builds': 0, 'excluded': True, 'auto_exclusion': 'transition',
                       'needs_review': False, 'review_reasons': [EXCLUSION_REASON]})
        transition_exclusions.append({'group': group_id, 'representative': item['id'],
                                      **item['transition']})
    transition_reviews = [
        {'group': group['id'], 'representative': group['representative'],
         **by_id[group['representative']]['transition_review']}
        for group in groups
        if group['representative'] in by_id and by_id[group['representative']].get('transition_review')
    ]
    audit_log['transition_detection'] = {'auto_excluded': transition_exclusions, 'needs_review': transition_reviews}
    # Reconcile suspicious pairs to final groups and retain the closest example per pair.
    owner = {m: g['id'] for g in groups for m in g['members']}
    unique_pairs = {}
    for c in candidates:
        ga, gb = owner[c['a']], owner[c['b']]
        if ga == gb:
            continue
        key = tuple(sorted((ga, gb)))
        if key not in unique_pairs or c['mae'] < unique_pairs[key]['mae']:
            unique_pairs[key] = dict(c, group_a=ga, group_b=gb)
    metadata = [{k: v for k, v in i.items() if not k.startswith('_') and k not in {'normalized','gray','edges','descriptor','position'}} for i in items]
    manifest = {'version': 2, 'program_version':APP_VERSION, 'source': str(source.path), 'build_policy': args.builds,
                'settings': {'crop': args.crop, 'workers': args.workers,
                             'ocr':getattr(args,'ocr','off'), 'min_long_edge':getattr(args,'min_long_edge',1280),
                             'min_short_edge':getattr(args,'min_short_edge',720),
                             'ocr_workers':getattr(args,'ocr_workers',4),
                             'ocr_cache':str(Path(getattr(args,'ocr_cache',Path(__file__).with_name('.cache')/'ocr')).resolve())},
                'runtime':{'python':platform.python_version(),'opencv':cv2.__version__,'numpy':np.__version__},'images': metadata,
                'groups': groups, 'decisions': decisions, 'review_pairs': sorted(unique_pairs.values(), key=lambda c:c['mae']),
                'errors': errors,'audit':audit_log}
    # Scope saved decisions to the recommendations as well as the source pixels.
    review_inputs = {
        'images': [(i['id'], i['sha256'], i['crop'], i.get('quad')) for i in metadata],
        'groups': groups,
        'pairs': manifest['review_pairs'],
    }
    if audit_log.get('numeric_reviews'):
        review_inputs['numeric_reviews'] = audit_log['numeric_reviews']
    manifest['manifest_id'] = hashlib.sha256(json.dumps(review_inputs, sort_keys=True).encode()).hexdigest()
    print('Writing native-resolution slides and text...', flush=True)
    audit_log['export_text'] = export(source, manifest, out / 'selected')
    elapsed = time.perf_counter() - start
    manifest['stats'] = {'input_images': len(source.names), 'processed': len(items), 'groups': len(groups),
                         'removed': len(items)-len(groups), 'review_pairs': len(unique_pairs),
                         'selected':sum(not g.get('excluded') for g in groups),
                         'needs_review':sum(g.get('needs_review',False) and not g.get('excluded') for g in groups),
                         'auto_excluded':sum(g.get('excluded',False) for g in groups),
                         'read_seconds': round(read_seconds, 2), 'total_seconds': round(elapsed, 2)}
    report(out, manifest)
    pending_manifest = out / 'manifest.tmp'
    pending_manifest.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    pending_manifest.replace(out / 'manifest.json')
    source.close()
    print(json.dumps(manifest['stats'], indent=2))
    print(f"Review: {out / 'review.html'}", flush=True)


def export(source, manifest, dest, overrides=None, crops=None):
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'originals').mkdir(exist_ok=True)
    (dest / 'slides').mkdir(exist_ok=True)
    (dest / 'text').mkdir(exist_ok=True)
    images = {i['id']: i for i in manifest['images']}
    groups = [g for g in (overrides if overrides is not None else manifest['groups']) if not g.get('excluded')]
    settings = manifest.get('settings', {})
    ocr_enabled = settings.get('ocr') == 'on'
    cache = settings.get('ocr_cache', Path(__file__).with_name('.cache') / 'ocr')
    workers = max(1, min(len(groups), settings.get('ocr_workers', 4))) if ocr_enabled else 1
    if ocr_enabled:
        from slide_ocr import read_text, refine_export_text

    def write_selected(task):
        seq, group = task
        item = images[group['representative']]
        override = (crops or {}).get(str(item['id']))
        if override is not None:
            item = {**item, 'crop': override, 'quad': None}
        data = source.read(item['name'])
        if item.get('sha256') and hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError(f'Source image changed since analysis: {item["name"]}. Re-run the scan.')
        (dest / 'originals' / f"{seq:04d}{Path(item['name']).suffix.lower()}").write_bytes(data)
        slide = extract(decode(data), item)
        ocr = item.get('ocr', {})
        if overrides is not None and ocr_enabled and (override is not None or ocr.get('status') != 'ok'):
            ocr = read_text(slide, cache)
        elif override is not None:
            ocr = {'status': 'crop-changed'}
        height, width = slide.shape[:2]
        footer = footer_number(ocr, width, height)
        if ocr_enabled and ocr.get('status') == 'ok':
            ocr = refine_export_text(slide, ocr, cache)
        ok, encoded = cv2.imencode('.png', slide)
        if not ok:
            raise ValueError('PNG encoding failed')
        (dest / 'slides' / f'{seq:04d}.png').write_bytes(encoded.tobytes())
        if overrides is None:
            item['native_preview'] = f'selected/slides/{seq:04d}.png'
        if ocr.get('status') == 'ok':
            (dest / 'text' / f'{seq:04d}.txt').write_text(ocr['text'], encoding='utf-8')
            (dest / 'text' / f'{seq:04d}.json').write_text(json.dumps(ocr, ensure_ascii=False, indent=2), encoding='utf-8')
        record = slide_text_record(seq, item['name'], slide, ocr)
        rows = [[seq, item['name'], images[member]['name'], *item['crop'], width, height,
                 '; '.join(group.get('review_reasons', [])), json.dumps(item.get('quad'))] for member in group['members']]
        return record, footer, rows, ocr.get('export_refinement')

    records, footers = [], []
    refinement = {'attempted': 0, 'applied': 0}
    with ((dest / 'mapping.csv').open('w', newline='', encoding='utf-8-sig') as f,
          (dest / SESSION_TEXT).open('w', encoding='utf-8', newline='\n') as session_text,
          concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool):
        writer = csv.writer(f)
        writer.writerow(['slide', 'selected_source', 'member_source', 'crop_x1', 'crop_y1', 'crop_x2', 'crop_y2',
                         'slide_width','slide_height','quality_flags','quad_json'])
        for record, footer, rows, refined in pool.map(write_selected, enumerate(groups, 1)):
            records.append(record)
            footers.append(footer)
            writer.writerows(rows)
            if refined:
                refinement['attempted'] += 1
                refinement['applied'] += int(refined['applied'])
        confirm_footer_numbers(records, footers)
        for record in records:
            session_text.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
    return refinement


def review_page(manifest):
    template = Path(__file__).with_name('review_template.html').read_text(encoding='utf-8')
    style = Path(__file__).with_name('app.css').read_text(encoding='utf-8')
    payload = json.dumps(manifest).replace('<', '\\u003c')
    return template.replace('__APP_STYLE__', style).replace('__MANIFEST__', payload)


def report(out, manifest):
    (out / 'review.html').write_text(review_page(manifest), encoding='utf-8')
    (out / 'review.js').write_text(Path(__file__).with_name('review.js').read_text(encoding='utf-8'),encoding='utf-8')


def validate_review(manifest,decision):
    valid={i['id'] for i in manifest['images']}
    image_info={i['id']:i for i in manifest['images']}
    seen=[]
    group_ids=[]
    for group in decision['groups']:
        if type(group['id']) is not int or group['id']<1:
            raise ValueError('Group IDs must be positive integers')
        if any(type(i) is not int for i in group['members']) or type(group['representative']) is not int:
            raise ValueError('Image IDs must be integers')
        if 'excluded' in group and type(group['excluded']) is not bool:
            raise ValueError('Excluded must be true or false')
        if not group['members'] or group['representative'] not in group['members']:
            raise ValueError('Each representative must belong to its nonempty group')
        seen.extend(group['members'])
        group_ids.append(group['id'])
    if set(seen)!=valid or len(seen)!=len(valid) or len(set(group_ids))!=len(group_ids):
        raise ValueError('Review must contain unique groups and every image exactly once')
    if decision.get('source')!=manifest['source']:
        raise ValueError('Review belongs to a different source')
    if manifest.get('manifest_id') and decision.get('manifest_id')!=manifest['manifest_id']:
        raise ValueError('Review belongs to a different scan')
    for key,box in decision.get('crops',{}).items():
        if int(key) not in valid or len(box)!=4 or not all(isinstance(x,(int,float)) and np.isfinite(x) for x in box):
            raise ValueError('Invalid per-image crop')
        if not (0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1):
            raise ValueError('Crop must have positive width/height within the image')
        item=image_info[int(key)]
        if round(box[2]*item['width'])-round(box[0]*item['width'])<2 or round(box[3]*item['height'])-round(box[1]*item['height'])<2:
            raise ValueError('Crop must contain at least two pixels in each dimension')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    scan = sub.add_parser('scan', help='Analyze ZIP or folder and export representative slides')
    scan.add_argument('input')
    scan.add_argument('--output', required=True)
    scan.add_argument('--crop', type=lambda s: tuple(float(x) for x in s.split(',')), help='Normalized x1,y1,x2,y2; default auto')
    scan.add_argument('--builds', choices=['complete','all'], default='complete')
    scan.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 4))
    scan.add_argument('--ocr',choices=['on','off'],default='on',help='Offline OCR verification and text export (default on)')
    scan.add_argument('--ocr-workers',type=int,default=min(4,os.cpu_count() or 2))
    scan.add_argument('--ocr-cache',default=str(Path(__file__).with_name('.cache')/'ocr'))
    scan.add_argument('--min-long-edge',type=int,default=1280)
    scan.add_argument('--min-short-edge',type=int,default=720)
    review = sub.add_parser('export', help='Apply downloaded review decisions')
    review.add_argument('manifest')
    review.add_argument('--decisions', required=True)
    review.add_argument('--output', required=True)
    serve=sub.add_parser('serve',help='Review at native resolution and export directly in the browser')
    serve.add_argument('output',help='Completed scan directory')
    serve.add_argument('--port',type=int,default=0,help='Loopback port; default chooses an available port')
    serve.add_argument('--open',action='store_true',help='Open the review page in your browser')
    args = parser.parse_args(argv)
    if args.command == 'scan':
        if args.crop and (len(args.crop) != 4 or not (0 <= args.crop[0] < args.crop[2] <= 1 and 0 <= args.crop[1] < args.crop[3] <= 1)):
            parser.error('--crop must be x1,y1,x2,y2 within 0..1 with positive width/height')
        if min(args.workers,args.ocr_workers,args.min_long_edge,args.min_short_edge)<1:
            parser.error('Worker counts and resolution thresholds must be positive')
        run(args)
    elif args.command=='serve':
        from review_server import serve as serve_report
        serve_report(args.output,args.port,args.open)
    else:
        manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
        decision = json.loads(Path(args.decisions).read_text(encoding='utf-8'))
        try:
            validate_review(manifest,decision)
        except (KeyError,ValueError,TypeError) as exc:
            parser.error(str(exc))
        dest = Path(args.output).resolve()
        if dest.exists() and any(dest.iterdir()):
            parser.error('Use a new, empty output directory')
        source = Source(manifest['source'])
        export(source, manifest, dest, decision['groups'],decision.get('crops'))
        source.close()
        (dest / 'review-decisions.json').write_text(json.dumps(decision, indent=2), encoding='utf-8')
        count = sum(not g.get('excluded') for g in decision['groups'])
        print(f'Exported {count} reviewed slides to {dest}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError,OSError,RuntimeError,zipfile.BadZipFile) as exc:
        import sys
        print(f'Error: {exc}',file=sys.stderr)
        raise SystemExit(1)
