# Build the Windows release

Use Windows x64 and 64-bit Python 3.13. To build from a fresh clone in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\build-windows.ps1
```

If PowerShell requires permission to run downloaded scripts, follow your organization’s policy or use an approved build runner. The GitHub Actions **Build Windows executable** workflow builds an artifact on demand.

The build writes `dist/ConferenceSlideExtractor-Windows-x64.exe` and `dist/SHA256SUMS.txt`. The executable bundles its Python runtime, review assets, and the OCR and YuNet models using PyInstaller. It launches a local browser app by default and accepts `scan`, `serve`, and `export` as command-line subcommands. [PyInstaller bundle documentation](https://pyinstaller.org/en/stable/operating-mode.html)

The Windows build runs the unit suite, verifies the bundled model and the CLI export, then tests a timestamped build and transition sequence. Headless browser checks cover ZIP intake, review, download, and exclusion recovery. To distribute a GitHub build, run **Build Windows executable** and create a draft release targeting the tested commit. Run **Attach Windows release assets** with the successful build run ID and the numeric draft release ID from the GitHub Releases API. It verifies the build and checksum before uploading the assets. A maintainer then publishes the draft. The repository’s automation token can upload assets but cannot publish this draft. Keep pilot ZIPs, exports, images, and caches outside Git.

Upstream components include [Python](https://docs.python.org/3/license.html), [OpenCV](https://opencv.org/license/), [NumPy](https://numpy.org/doc/stable/license.html), [Pillow](https://pillow.readthedocs.io/en/stable/about.html), [ONNX Runtime](https://github.com/microsoft/onnxruntime/blob/main/LICENSE), and [RapidOCR and its model notices](https://github.com/RapidAI/RapidOCR/blob/main/python/MODEL_LICENSES.md). The build collects package metadata for the principal bundled dependencies.
