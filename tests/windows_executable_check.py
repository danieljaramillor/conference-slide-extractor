"""Run the frozen application from an empty folder using its bundled OCR."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import cv2

from mixed_fixtures import fixtures
from scan_integrity_check import check as check_integrity
from speaker_fixtures import camera_capture


def check(executable, browser=False):
    with tempfile.TemporaryDirectory(prefix='slide-exe-check-') as directory:
        root = Path(directory)
        archive = root / 'example.zip'
        cases = fixtures()
        cases.append(('speaker-only', camera_capture(), None))
        with zipfile.ZipFile(archive, 'w') as zipped:
            for name, image, _ in cases:
                success, encoded = cv2.imencode('.png', image)
                assert success
                zipped.writestr(f'{name}.png', encoded.tobytes())
        output = root / 'results'
        subprocess.run([str(executable), 'scan', str(archive), '--output', str(output), '--workers', '2',
                        '--ocr-workers', '2', '--ocr-cache', str(root / 'cache')], cwd=root, check=True, timeout=120)
        manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        assert manifest['stats']['processed'] == len(cases) and not manifest['errors']
        assert manifest['stats']['selected'] == 4
        assert manifest['audit']['ocr_errors'] == 0
        detection = manifest['audit']['speaker_detection']
        assert detection['status'] == 'ok'
        speaker = next(image['id'] for image in manifest['images'] if image['name'] == 'speaker-only.png')
        assert [entry['representative'] for entry in detection['auto_excluded']] == [speaker]
        assert check_integrity(output) == {'captures': len(cases), 'slides': 4, 'text_files': 4}
        assert (output / 'review.html').is_file()
        if browser:
            from browser_batch_check import check as check_browser
            from browser_exclusion_check import check as check_exclusions

            check_browser(executable, sys.executable, str(archive), root / 'dashboard.png', browser_channel=None)
            check_exclusions(output, browser_channel=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('--browser', action='store_true', help='Use Playwright Chromium for the packaged ZIP-to-export flow')
    args = parser.parse_args()
    check(args.executable.resolve(), args.browser)
