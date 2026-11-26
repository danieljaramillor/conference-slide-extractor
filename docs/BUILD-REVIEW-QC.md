# Grouped build review QA

## Original conference pilot

The 1,450-capture pilot produced 46 suggested comparisons across 12 build groups. All 17 established complete builds retained their earlier export selections. These include a late results panel and a late QR code. The 135 selected native PNGs and their 135 OCR text and 135 OCR JSON files matched v2.8 byte for byte.

The comparison endpoint produced registered highlights on 34 of the 46 suggested pairs when tested directly. The remaining 12 had insufficient alignment evidence and stayed available for side-by-side review. The deck inspector displayed the selected results-panel capture against the earlier stage and highlighted the panel, with no edit to the export selection.

## ESMO Screenshots sample

A test ZIP contained 60 captures from three SharePoint folders, with 18 from each of two poster sequences and 24 from a conference session. The scan processed all 60 captures and initially selected 10 slides, each with a native PNG and OCR output. The session included a nine-capture build with two solid reveals and a partially faded reveal. The selected capture preserved the complete workflow diagram. The sample also surfaced a presenter-only change outside the slide pane and a separate pair with different slide content. Use **Adjust crop** in the comparison to isolate the slide in both images when the video pane is included.

These figures describe the sampled ZIP. Source screenshots, text, and test exports are kept with the local QA data.

## Repeat the checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m playwright install chromium --only-shell
.\.venv\Scripts\python.exe tests\windows_executable_check.py .\dist\ConferenceSlideExtractor-Windows-x64.exe --browser
```

The browser checks cover related variants, image highlights, additive and replacement content, merge choices, previous decisions, undo, reload, and ZIP export. A controlled export checks the retained PNG pixels against the original capture. To check a fresh scan of the original pilot, pass the established baseline and fresh result directories to `tests/pilot_build_check.py`, then pass the fresh result directory to `tests/scan_integrity_check.py`.
