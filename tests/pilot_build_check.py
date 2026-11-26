"""Check established build representatives against a fresh scan of the same pilot."""

import argparse
import json
from pathlib import Path


def exported_paths(output, manifest):
    selected = [group for group in manifest['groups'] if not group.get('excluded')]
    return {group['representative']: (
        output / 'selected' / 'slides' / f'{index:04d}.png',
        output / 'selected' / 'text' / f'{index:04d}.txt',
    ) for index, group in enumerate(selected, 1)}


def check(baseline, fresh):
    before = json.loads((baseline / 'manifest.json').read_text(encoding='utf-8'))
    after = json.loads((fresh / 'manifest.json').read_text(encoding='utf-8'))
    old_images = {(image['name'], image['sha256']): image['id'] for image in before['images']}
    new_images = {(image['name'], image['sha256']): image['id'] for image in after['images']}
    if old_images.keys() != new_images.keys():
        raise AssertionError('Baseline and fresh scan must use the same input captures')
    new_ids = {previous: new_images[key] for key, previous in old_images.items()}
    old_paths, new_paths = exported_paths(baseline, before), exported_paths(fresh, after)
    old_builds = [group for group in before['groups'] if group['builds'] and not group.get('excluded')]
    owners = {member: group for group in after['groups'] for member in group['members']}
    if not old_builds:
        raise AssertionError('Baseline has no build groups to compare')

    result = {'build_representatives': [], 'exact_native_pngs': 0, 'exact_ocr_text_files': 0}
    for group in old_builds:
        old_id = group['representative']
        new_id = new_ids[old_id]
        if new_id not in new_paths:
            raise AssertionError(f'Build representative {old_id} was not selected in the fresh scan')
        for index, key in enumerate(('exact_native_pngs', 'exact_ocr_text_files')):
            if old_paths[old_id][index].read_bytes() != new_paths[new_id][index].read_bytes():
                raise AssertionError(f'{key} changed for build representative {old_id}')
            result[key] += 1
        result['build_representatives'].append(old_id)

    for earlier, complete in ((274, 278), (1373, 1374)):
        group = owners[new_ids[earlier]]
        if group['representative'] != new_ids[complete] or group.get('excluded'):
            raise AssertionError(f'Expected late reveal {complete} to represent the earlier stage {earlier}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path, help='Completed scan of the original 1,450-frame pilot')
    parser.add_argument('fresh', type=Path, help='A new scan of the same pilot')
    args = parser.parse_args()
    print(json.dumps(check(args.baseline, args.fresh), indent=2))
