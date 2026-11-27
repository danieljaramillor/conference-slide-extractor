"""Fail the build if PyInstaller silently omits a required runtime or asset."""
from pathlib import PureWindowsPath
import sys

from PyInstaller.archive.readers import CArchiveReader


def check(path):
    names = {PureWindowsPath(entry).name.lower() for entry in CArchiveReader(path).toc}
    required = {
        '_tkinter.pyd', 'init.tcl', 'tk.tcl', 'tcl86t.dll', 'tk86t.dll',
        'msvcp140.dll', 'msvcp140_1.dll', 'vcruntime140.dll', 'vcruntime140_1.dll',
        'onnxruntime.dll', 'pp-ocrv6_det_small.onnx', 'pp-ocrv6_rec_small.onnx',
        'ch_ppocr_mobile_v2.0_cls_mobile.onnx', 'face_detection_yunet_2023mar.onnx',
        'batch_app.html', 'review_template.html', 'review.js', 'app.css',
        'practice.zip', 'server-guide.html',
    }
    if missing := required - names:
        raise SystemExit('Missing bundle files: ' + ', '.join(sorted(missing)))


if __name__ == '__main__':
    check(sys.argv[1])
