# Grouped build review QA

## Original conference pilot

The 1,450-capture pilot produced 46 suggested comparisons across 12 build groups. All 17 established complete builds retained their earlier export selections. These include a late results panel and a late QR code. The 135 selected native PNGs and their 135 OCR text and 135 OCR JSON files matched v2.8 byte for byte.

The comparison endpoint produced registered highlights on 34 of the 46 suggested pairs when tested directly. The remaining 12 had insufficient alignment evidence and stayed available for side-by-side review. The deck inspector displayed the selected results-panel capture against the earlier stage and highlighted the panel, with no edit to the export selection.

## ESMO Screenshots sample

A test ZIP contained 60 captures from three SharePoint folders, with 18 from each of two poster sequences and 24 from a conference session. In v2.9 the scan initially selected 10 slides and raised two suggested comparisons. Presenter motion caused one duplicate.

## Automatic presenter cropping in v2.10

| ZIP | Captures | Result |
| --- | --- | --- |
| Original ESMO sample, three folders | 60 | All 24 session captures automatically cropped to 1429 × 804 native pixels. Nine selected slides and zero suggested comparisons or slide issues. The complete workflow build and the separate disclosure slide both stayed selected. |
| Expanded ESMO sample, six additional folders | 118 | 24 selected slides. The scan recovered one missed crop using its adjacent captures, grouped the duplicate, and retained the chart with both late outlines and a separate build with its full second chart. Two ambiguous frames remained at full size and were flagged in **Slide issues**. |
| Original conference pilot | 1,450 | All 17 previously verified complete-build selections stayed selected. The same 135 native slide PNGs and their 135 OCR text and 135 OCR JSON files matched v2.9 byte for byte. |

The expanded sample includes two posters, three oral presentations, and a longer session. The representative with both chart outlines came from the final solid reveal. A separate scan using `--builds all` retained the plain chart and both successive outline reveals. The three oral presentations kept their native crops, including an example with three successive reveals and its distinct following slide. One repeated poster table still appeared in multiple groups after OCR split adjacent values.

The crop detector uses stable framing and independent presenter motion across a sequence, or a consistent pane confirmed by two adjacent captures. It keeps the full source capture alongside the crop. Tests cover both pane positions, static layouts, an interruption in capture order, unrelated broadcast framing, and a portrait that belongs to the slide itself.

These figures describe the sampled ZIP. Source screenshots, text, and test exports are kept with the local QA data.

## Repeat the checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m playwright install chromium --only-shell
.\.venv\Scripts\python.exe tests\windows_executable_check.py .\dist\ConferenceSlideExtractor-Windows-x64.exe --browser
```

The browser checks cover related variants, image highlights, additive and replacement content, merge choices, previous decisions, undo, reload, and ZIP export. A controlled export checks the retained PNG pixels against the original capture. To check a fresh scan of the original pilot, pass the established baseline and fresh result directories to `tests/pilot_build_check.py`, then pass the fresh result directory to `tests/scan_integrity_check.py`.
