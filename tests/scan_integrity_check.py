"""Verify scan and export files against a completed batch and its original captures."""
import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slide_dedupe import Source, decode, extract, validate_review


def check(batch, export_directory=None):
    manifest = json.loads((batch / 'manifest.json').read_text(encoding='utf-8'))
    output = export_directory if export_directory is not None else batch / 'selected'
    decision_path = output / 'review-decisions.json'
    decision = (json.loads(decision_path.read_text(encoding='utf-8')) if decision_path.exists() else
                {key: manifest[key] for key in ('source', 'manifest_id', 'groups')})
    validate_review(manifest, decision)
    images = {image['id']: image for image in manifest['images']}
    groups = [group for group in decision['groups'] if not group.get('excluded')]
    crops = decision.get('crops', {})
    originals, slides, text_files = set(), set(), set()

    source = Source(manifest['source'])
    text_count = 0
    try:
        with (output / 'mapping.csv').open(encoding='utf-8-sig', newline='') as mapping:
            rows = iter(csv.DictReader(mapping))
            for position, group in enumerate(groups, 1):
                item = images[group['representative']]
                override = crops.get(str(item['id']))
                if override is not None:
                    item = {**item, 'crop': override, 'quad': None}
                sequence = f'{position:04d}'
                original_name = sequence + Path(item['name']).suffix.lower()
                slide_name = sequence + '.png'
                originals.add(original_name)
                slides.add(slide_name)
                original = source.read(item['name'])
                assert (output / 'originals' / original_name).read_bytes() == original, original_name
                native = extract(decode(original), item)
                png = (output / 'slides' / slide_name).read_bytes()
                rendered = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
                np.testing.assert_array_equal(rendered, native, err_msg=slide_name)
                for member in group['members']:
                    row = next(rows)
                    assert int(row['slide']) == position
                    assert row['selected_source'] == item['name']
                    assert row['member_source'] == images[member]['name']
                    assert (int(row['slide_height']), int(row['slide_width'])) == native.shape[:2]
                    assert [float(row[key]) for key in ('crop_x1', 'crop_y1', 'crop_x2', 'crop_y2')] == item['crop']
                text_path = output / 'text' / (sequence + '.txt')
                if not decision_path.exists() and item.get('ocr', {}).get('status') == 'ok':
                    assert text_path.is_file(), sequence
                if text_path.is_file():
                    record = text_path.with_suffix('.json')
                    text = json.loads(record.read_text(encoding='utf-8'))
                    assert text_path.read_text(encoding='utf-8') == text['text'], sequence
                    text_files.update((text_path.name, record.name))
                    text_count += 1
            assert next(rows, None) is None, 'Unexpected rows after the final selected group'
    finally:
        source.close()

    assert {path.name for path in (output / 'originals').iterdir()} == originals
    assert {path.name for path in (output / 'slides').iterdir()} == slides
    assert {path.name for path in (output / 'text').iterdir()} == text_files
    return {'captures': len(images), 'slides': len(slides), 'text_files': text_count}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--export-directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.batch, args.export_directory)))
