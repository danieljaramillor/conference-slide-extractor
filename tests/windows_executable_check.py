"""Run the frozen application from an empty folder using its bundled OCR."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import cv2
import numpy as np

from mixed_fixtures import compact_table_captures, fixtures
from presenter_fixtures import SLIDE_BOUNDS, presenter_annotation_sequence, presenter_capture_sequence
from scan_integrity_check import check as check_integrity
from speaker_fixtures import camera_capture
from transition_fixtures import build_transition_sequence, one_sided_sequence, timestamped


def scan_captures(executable, root, stem, cases):
    archive = root / f'{stem}.zip'
    with zipfile.ZipFile(archive, 'w') as zipped:
        for name, image in cases:
            success, encoded = cv2.imencode(Path(name).suffix, image)
            assert success
            zipped.writestr(name, encoded.tobytes())
    output = root / f'{stem}-results'
    subprocess.run([str(executable), 'scan', str(archive), '--output', str(output), '--workers', '2',
                    '--ocr-workers', '2', '--ocr-cache', str(root / 'cache')], cwd=root, check=True, timeout=120)
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['stats']['processed'] == len(cases) and not manifest['errors']
    return archive, output, manifest


def check_transitions(executable, root):
    cases = build_transition_sequence()
    _, output, manifest = scan_captures(executable, root, 'transitions', cases)
    assert [group['representative'] for group in manifest['groups'] if not group.get('excluded')] == [2, 4]
    assert [group['representative'] for group in manifest['groups'] if group.get('auto_exclusion') == 'transition'] == [3]
    integrity = check_integrity(output)
    assert integrity['captures'] == len(cases) and integrity['slides'] == 2
    return output


def check_one_sided(executable, root):
    cases, outgoing = one_sided_sequence()
    _, output, manifest = scan_captures(executable, root, 'one-sided', cases)
    groups = [group for group in manifest['groups'] if not group.get('excluded')]
    assert [group['representative'] for group in groups] == [0, 2, 3]
    blend = groups[1]
    assert blend['members'] == [2] and 'possible blend with unique content' in blend['review_reasons']
    assert manifest['images'][2]['transition_review']['after'] == 3
    assert not manifest['review_pairs']
    assert 'Participating centers received support for data collection' in (output / 'selected/text/0002.txt').read_text(encoding='utf-8')
    assert check_integrity(output) == {'captures': len(cases), 'slides': 3, 'text_files': 3}
    extra = timestamped([image for _, image in cases] + [outgoing, outgoing])
    _, resolved, verified = scan_captures(executable, root, 'one-sided-outgoing-found', extra)
    assert [group['representative'] for group in verified['groups'] if not group.get('excluded')] == [0, 3, 5]
    exclusions = verified['audit']['transition_detection']['auto_excluded']
    assert [(entry['representative'], entry['kind'], entry['before']) for entry in exclusions] == [(2, 'one-sided-crossfade', 5)]
    assert check_integrity(resolved) == {'captures': len(extra), 'slides': 3, 'text_files': 3}
    return output


def check_presenter_crops(executable, root):
    cases = presenter_capture_sequence()
    _, output, manifest = scan_captures(executable, root, 'presenter-crops', cases)
    selected = [group for group in manifest['groups'] if not group.get('excluded')]
    assert len(selected) == 3, selected
    build = next(group for group in selected if 0 in group['members'])
    assert build['members'] == list(range(7)) and build['representative'] in (5, 6), build
    assert {group['representative'] for group in selected if group is not build} == {7, 8}, selected
    assert not manifest['review_pairs'], manifest['review_pairs']
    assert manifest['audit']['presenter_crop']['captures'] == 8

    left, top, right, bottom = SLIDE_BOUNDS
    for image in manifest['images'][:8]:
        pixel_box = [round(value * axis) for value, axis in zip(image['crop'], (1920, 1080, 1920, 1080))]
        assert pixel_box == list(SLIDE_BOUNDS) and image['quad'] is None, image
        assert image['crop_mode'] == 'presenter-slide', image
        assert (image['slide_width'], image['slide_height']) == (right - left, bottom - top), image
    assert manifest['images'][8]['crop'] == [0.0, 0.0, 1.0, 1.0]

    for export_index, group in enumerate(selected, 1):
        capture_index = group['representative']
        _, frame = cases[capture_index]
        expected = frame[top:bottom, left:right] if capture_index < 8 else frame
        exported = cv2.imread(str(output / 'selected/slides' / f'{export_index:04d}.png'))
        np.testing.assert_array_equal(exported, expected)
    build_text = (output / 'selected/text' / f'{selected.index(build) + 1:04d}.txt').read_text(encoding='utf-8')
    assert 'Trial results' in build_text and '47.8' in build_text, build_text
    assert 'Broadcast Hall North' not in build_text and 'Presenter information' not in build_text, build_text
    assert check_integrity(output) == {'captures': len(cases), 'slides': 3, 'text_files': 3}


def check_presenter_annotations(executable, root):
    cases = presenter_annotation_sequence()
    _, output, manifest = scan_captures(executable, root, 'presenter-annotations', cases)
    assert manifest['audit']['presenter_crop']['captures'] == len(cases)
    assert not manifest['review_pairs'], manifest['review_pairs']
    assert len(manifest['groups']) == 1, manifest['groups']
    group, = manifest['groups']
    assert group['representative'] in (4, 5), group
    assert group['members'] == list(range(len(cases))) and group['builds'], group

    left, top, right, bottom = SLIDE_BOUNDS
    exported = cv2.imread(str(output / 'selected/slides/0001.png'))
    np.testing.assert_array_equal(exported, cases[4][1][top:bottom, left:right])
    text = (output / 'selected/text/0001.txt').read_text(encoding='utf-8')
    assert 'Treatment response by cohort' in text, text
    assert check_integrity(output) == {'captures': len(cases), 'slides': 1, 'text_files': 1}


def check_compact_table(executable, root):
    cases = compact_table_captures()
    _, output, manifest = scan_captures(executable, root, 'compact-table', cases)
    groups = manifest['groups']
    assert [sorted(group['members']) for group in groups] == [[0, 1, 2], [3]], groups
    assert not manifest['review_pairs'], manifest['review_pairs']
    splits = manifest['audit']['numeric_splits']
    assert [split['a'] for split in splits] == [3], splits
    assert any('42.8' in first and '47.8' in second for first, second in splits[0]['numeric_changes']), splits

    for index, expected in enumerate(('47.8', '42.8'), 1):
        text = (output / 'selected/text' / f'{index:04d}.txt').read_text(encoding='utf-8')
        assert 'Outcomes by treatment group' in text and expected in text, text
    assert check_integrity(output) == {'captures': len(cases), 'slides': 2, 'text_files': 2}


def check(executable, browser=False):
    with tempfile.TemporaryDirectory(prefix='slide-exe-check-') as directory:
        root = Path(directory)
        cases = [(f'{name}.png', image) for name, image, _ in fixtures()]
        cases.append(('speaker-only.png', camera_capture()))
        archive, output, manifest = scan_captures(executable, root, 'example', cases)
        assert manifest['stats']['selected'] == 4
        assert manifest['audit']['ocr_errors'] == 0
        detection = manifest['audit']['speaker_detection']
        assert detection['status'] == 'ok'
        speaker = next(image['id'] for image in manifest['images'] if image['name'] == 'speaker-only.png')
        assert [entry['representative'] for entry in detection['auto_excluded']] == [speaker]
        assert check_integrity(output) == {'captures': len(cases), 'slides': 4, 'text_files': 4}
        assert (output / 'review.html').is_file()
        transition_output = check_transitions(executable, root)
        one_sided_output = check_one_sided(executable, root)
        check_presenter_crops(executable, root)
        check_presenter_annotations(executable, root)
        check_compact_table(executable, root)
        if browser:
            from browser_batch_check import check as check_browser
            from browser_exclusion_check import check as check_exclusions, check_retained_blend, check_transition_review
            from browser_group_check import check as check_group_review

            check_browser(executable, sys.executable, str(archive), root / 'dashboard.png', browser_channel=None)
            check_exclusions(output, browser_channel=None)
            check_transition_review(transition_output, browser_channel=None)
            check_retained_blend(one_sided_output, browser_channel=None)
            check_group_review(executable=executable, browser_channel=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('--browser', action='store_true', help='Use Playwright Chromium for the packaged ZIP-to-export flow')
    args = parser.parse_args()
    check(args.executable.resolve(), args.browser)
