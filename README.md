# Conference Slide Extractor

Turn a ZIP of conference screenshots into readable slide images and text. The Windows app detects slide regions, groups repeat captures, and opens possible duplicates and slide builds side by side for review. Processing and OCR run on your computer.

The app excludes clear speaker-only camera shots and confirmed blends between slides from the slide export. When a blend may hold the only record of an earlier slide, the app preserves it and flags it in **Slide issues**. **Review excluded captures** lets an operator inspect and restore an automatic exclusion.

For recurring broadcast layouts with a separate presenter pane, the app uses neighboring captures to verify the slide boundary. It crops out the presenter and broadcast graphics before matching slides or reading text. The export keeps the full source screenshot alongside each native-resolution slide image.

When a slide reveals more information, the app looks for the capture that contains the earlier content and the new material. It checks small text and table differences against the native capture before grouping slides. Rescan the source ZIP after upgrading to apply the latest checks to an existing batch.

## Install on Windows

1. Open [GitHub Releases](https://github.com/danieljaramillor/conference-slide-extractor/releases/latest). Sign in with an account that has access to this repository.
2. Under **Assets**, download **ConferenceSlideExtractor-Windows-x64.exe**. Save it somewhere you can find it again, such as Downloads or a team tools folder.
3. Double-click the `.exe`. The app opens in your default browser. Keep the app window open while it processes or exports a batch.

The executable bundles Python, the OCR models, and their dependencies. It runs without an installer, API key, paid service, or GPU. The release executable is unsigned. Your organization’s device policy may require approval to run it. You can verify the download against `SHA256SUMS.txt` on the release page.

## Process and review a batch

1. Drop a screenshot ZIP into the dashboard or select **Choose ZIP**. Keep each session’s screenshots together with their original timestamped or consecutively numbered filenames. Select **Extract slides**, follow its progress in **Batches**, then select **Review slides**.
2. In **Review builds**, open a group of related captures and pick two to compare. **Highlight changes** marks where content differs when the image alignment is reliable. For a duplicate or additive build, choose **Merge, keep base** or **Merge, keep candidate** to retain the better capture. Use **Keep both slides** when each contains unique information. Click either image to enlarge it, or use **Undo last pair decision** to reverse a choice.
3. Continue to **Slide issues**. Enlarge a flagged slide, adjust the crop, or choose another capture. For **Check possible changed value**, compare the two native image crops and their OCR readings. Both slides stay in the deck when you select **Mark checked and next**. If the pixels show the same value, you can merge them in **Review deck**.
4. Continue to **Review deck** to check the included slides in export order. Use **Inspect grouped build** to compare the selected slide with an earlier capture. Use the **Excluded** filter to restore a slide. Edits that create new issues send you back to the relevant check before export.
5. Select **Continue to export**, then **Export ZIP…** and choose the folder and filename. If your browser shows **Export slides**, select **Download export ZIP** when it’s ready. The archive contains native-resolution PNGs, original captures, recognized text, line coordinates, and a CSV mapping. Open the same app and batch later to continue a saved review.

**Capture quality matters.** The app preserves available source pixels. Small, blurry, or compressed text can still be misread. Verify critical numbers against the original slide. If different chart or table frames contain unique data, keep each frame unless one capture includes all the content.

### Session text and tables

Each scan and reviewed ZIP includes `slides.jsonl`, a UTF-8 text file with one JSON record per included slide. `slide_number` is its one-based export position and matches its filename and `mapping.csv`. `printed_slide_number` is filled when OCR recognizes a slide or page label near the edge, or when adjacent slides show an incrementing number in the same footer position. The record also includes `source`, `ocr_status`, the full recognized `text`, and `tables`.

Export checks selected native images for small text and selectively runs a higher-resolution OCR pass. Tables store rows as arrays of text cells in reading order. The app uses visible grid lines and consistent alignment of recognized labels, headers, and values. Aligned tables include `cell_bounds` for cells with source coordinates. If a text box crosses a grid line, the app attempts to read the affected cells individually. `review_cells` identifies unresolved cells by one-based row and column and pixel bounds on the exported PNG. Those cells contain `null`. `cell_sources` identifies cells recovered by a local reread. `partial` and `inferred_rows` indicate where the structure or coverage merits a check. The per-slide `.txt` and OCR coordinate files remain in `text/`.

When a survival chart has a supported risk-count strip, the app checks its labels and time-axis positions and rereads each count from a separate crop. The resulting `risk_count_strip` table includes the time points, cell coordinates and any counts that still need review.

## Where files are stored

The app stores per-batch inputs, results, review decisions, logs, and OCR cache under `%LOCALAPPDATA%\Conference Slide Extractor\batches`. Paste that path into File Explorer to find it. Keep the same Windows user and data folder when replacing the executable with a new release. To choose a different data folder, run the executable with `--data`.

All review traffic stays on the computer running the app. The browser connects to the app’s local address. Pillow, OpenCV, and NumPy decode, locate, and compare slides. RapidOCR uses bundled PP-OCRv6 neural models and ONNX Runtime on the CPU. OpenCV uses the bundled YuNet model with slide and text checks to detect speaker-only captures. [YuNet model and license](assets/yunet/README.md)

## Windows server and automation

Use the same `.exe` on a Windows x64 processing machine. A remote operator can use an approved remote desktop connection or forward the loopback port through SSH. Start with one batch and verify the server’s storage permissions, browser access, and OCR before moving live work.

```powershell
.\ConferenceSlideExtractor-Windows-x64.exe --data 'D:\ConferenceBatches' --port 8765 --no-browser
```

For a job runner, scan an already complete archive directly. Use a fresh output directory per attempt and read `manifest.json` after a successful scan.

```powershell
.\ConferenceSlideExtractor-Windows-x64.exe scan 'D:\Incoming\session.zip' --output 'D:\Results\session-001'
```

Find the session text file at `D:\Results\session-001\selected\slides.jsonl`.

See [review and troubleshooting](docs/REVIEW.md), [grouped build QC](docs/BUILD-REVIEW-QC.md), [transition and build QC](docs/TRANSITION-QC.md), [speaker exclusion QC](docs/SPEAKER-QC.md), [build instructions](docs/BUILD.md), and the [scaling plan](SCALING.md).
