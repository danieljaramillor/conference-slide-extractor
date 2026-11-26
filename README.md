# Conference Slide Extractor

Turn a ZIP of conference screenshots into readable slide images and text. The Windows app detects slide regions, groups repeat captures, and opens possible duplicates and slide builds side by side for review. Processing and OCR run on your computer.

## Install on Windows

1. Open [GitHub Releases](https://github.com/danieljaramillor/conference-slide-extractor/releases/latest). Sign in with an account that has access to this repository.
2. Under **Assets**, download **ConferenceSlideExtractor-Windows-x64.exe**. Save it somewhere you can find it again, such as Downloads or a team tools folder.
3. Double-click the `.exe`. The app opens in your default browser. Keep the app window open while it processes or exports a batch.

The executable bundles Python, the OCR models, and their dependencies. It runs without an installer, API key, paid service, or GPU. The release executable is unsigned. Your organization’s device policy may require approval to run it. You can verify the download against `SHA256SUMS.txt` on the release page.

## Process and review a batch

1. Choose a ZIP of screenshots and select **Extract slides**. Progress appears in **Batches**. Select **Review slides** when it finishes.
2. In **Compare pairs**, inspect the two images. Choose **Keep both slides** if either contains unique information. For a duplicate or an additive build, select **Merge, keep left** or **Merge, keep right** to retain the better or more complete capture. Click either image to enlarge it. Use **Next** to skip a pair and **Undo last pair decision** to reverse a choice.
3. In **Slide issues**, check low-resolution or unclear crops. Open **All slides** to browse the full set, inspect alternate captures, or restore a slide from the **Excluded** filter.
4. Select **Export slides**, then **Download export ZIP**. It contains native-resolution PNGs, original captures, recognized text, line coordinates, and a CSV mapping. Open the same app and batch later to continue a saved review.

**Capture quality matters.** The app preserves available source pixels. Small, blurry, or compressed text can still be misread. Verify critical numbers against the original slide. If different chart or table frames contain unique data, keep each frame unless one capture includes all the content.

## Where files are stored

The app stores per-batch inputs, results, review decisions, logs, and OCR cache under `%LOCALAPPDATA%\Conference Slide Extractor\batches`. Paste that path into File Explorer to find it. Keep the same Windows user and data folder when replacing the executable with a new release. To choose a different data folder, run the executable with `--data`.

All review traffic stays on the computer running the app. The browser connects to the app’s local address. Pillow, OpenCV, and NumPy decode, locate, and compare slides. RapidOCR uses bundled PP-OCRv6 neural models and ONNX Runtime on the CPU to read text and help verify matches.

## Windows server and automation

Use the same `.exe` on a Windows x64 processing machine. A remote operator can use an approved remote desktop connection or forward the loopback port through SSH. Start with one batch and verify the server’s storage permissions, browser access, and OCR before moving live work.

```powershell
.\ConferenceSlideExtractor-Windows-x64.exe --data 'D:\ConferenceBatches' --port 8765 --no-browser
```

For a job runner, scan an already complete archive directly. Use a fresh output directory per attempt and read `manifest.json` after a successful scan.

```powershell
.\ConferenceSlideExtractor-Windows-x64.exe scan 'D:\Incoming\session.zip' --output 'D:\Results\session-001'
```

See [review and troubleshooting](docs/REVIEW.md), [build instructions](docs/BUILD.md), [pilot QC](QUALITY-CONTROL.md), and the [scaling plan](SCALING.md).
