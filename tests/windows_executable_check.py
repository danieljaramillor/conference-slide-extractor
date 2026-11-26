"""Run the frozen application from an empty folder using its bundled OCR."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import cv2

from mixed_fixtures import fixtures


def check(executable):
    with tempfile.TemporaryDirectory(prefix='slide-exe-check-') as directory:
        root = Path(directory)
        archive = root / 'example.zip'
        with zipfile.ZipFile(archive, 'w') as zipped:
            for name, image, _ in fixtures():
                success, encoded = cv2.imencode('.png', image)
                assert success
                zipped.writestr(f'{name}.png', encoded.tobytes())
        output = root / 'results'
        subprocess.run([str(executable), 'scan', str(archive), '--output', str(output), '--workers', '2',
                        '--ocr-workers', '2', '--ocr-cache', str(root / 'cache')], cwd=root, check=True, timeout=120)
        manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        assert manifest['stats']['selected'] == 4
        assert manifest['audit']['ocr_errors'] == 0
        assert len(list((output / 'selected' / 'text').glob('*.txt'))) == 4
        assert (output / 'review.html').is_file()


if __name__ == '__main__':
    check(Path(sys.argv[1]).resolve())
