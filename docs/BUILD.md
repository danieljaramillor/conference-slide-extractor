# Build the Windows release

Use Windows x64 and 64-bit Python 3.13. To build from a fresh clone in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\build-windows.ps1
```

If PowerShell requires permission to run downloaded scripts, follow your organization’s policy or use an approved build runner. The GitHub Actions **Build Windows executable** workflow builds an artifact on demand.

The build writes `dist/ConferenceSlideExtractor-Windows-x64.exe` and `dist/SHA256SUMS.txt`. The executable bundles its Python runtime, review assets, and OCR weights using PyInstaller. It launches a local browser app by default and accepts `scan`, `serve`, and `export` as command-line subcommands. [PyInstaller bundle documentation](https://pyinstaller.org/en/stable/operating-mode.html)

Before publishing a release, run a fresh mixed-layout ZIP through the compiled `.exe` and verify download of slides and OCR text. To publish through GitHub Actions, create a draft release for the version and run **Build Windows executable** with that tag in `release_tag`. The workflow tests the executable before uploading it and its checksum and publishing the draft. Leave `release_tag` blank for a build artifact only. Keep pilot ZIPs, exports, images, and caches outside Git.

Upstream components include [Python](https://docs.python.org/3/license.html), [OpenCV](https://opencv.org/license/), [NumPy](https://numpy.org/doc/stable/license.html), [Pillow](https://pillow.readthedocs.io/en/stable/about.html), [ONNX Runtime](https://github.com/microsoft/onnxruntime/blob/main/LICENSE), and [RapidOCR and its model notices](https://github.com/RapidAI/RapidOCR/blob/main/python/MODEL_LICENSES.md). The build collects package metadata for the principal bundled dependencies.
