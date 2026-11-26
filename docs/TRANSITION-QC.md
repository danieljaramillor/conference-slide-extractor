# Transition and slide build QA

The app checks nearby captures for an overlap between distinct slides. It looks for visual content from both ends of the transition, including a resized pane when it can align it reliably. For a dissolve across several captures, each intermediate frame must fit the same endpoints and follow the progression of the blend. A confirmed blend stays in **Review excluded captures** and is omitted from matching and export. Keeping it out of matching prevents it from joining two distinct slides into one group.

The app checks capture order within each folder using `image_HH-MM-SS` timestamps or consecutive numeric names such as `frame_102`. A gap, a change of folder, or a missing clean endpoint reduces the evidence available for an automatic decision.

## Conference pilot

The independent audit compared the new scan with Windows v2.6.0 on the same 1,450 captures.

| Check | Result |
| --- | --- |
| Capture integrity | All 1,450 source hashes and group memberships accounted for |
| Confirmed blends automatically omitted | Seven, including both previously missed patterns |
| Effect on selected output | Six fewer selected images. The seventh blend was already a non-representative capture |
| Unintended exclusions | Zero in the audited pilot |
| Established complete builds | All 17 representatives retained with byte-identical native PNG and OCR text files |
| Earlier speaker-only exclusions | All eight retained as exclusions |

The complete result-panel and QR captures stayed selected for their respective builds. All 135 retained PNG and OCR files matched the baseline byte for byte. The team also verified that real stage projections and distinct chart values stayed eligible. One visible disclosure blend was retained for pair review because its clean outgoing slide was absent from the screenshots. Omitting the blended frame could remove the only record of that slide.

## Re-run the checks

The regression tests cover bright and dark slides, late result panels and QR codes, changing capture layouts, different chart values, and genuine replacement of slide content. A four-frame dissolve case exports both clean endpoints. A separate four-stage reveal exports the complete version with the default policy and all solid stages with `--builds all`. An independent detector stress test covered 35 transformed blend cases and 85 slide controls under JPEG compression, reduced resolution, and contrast changes. All 35 blends were detected, with zero false exclusions among the controls. Windows CI exercises the packaged scan and browser export on a six-frame sequence with a build and a blend.

For the supplied pilot, compare two completed scan directories and verify the final files:

```powershell
.\.venv\Scripts\python.exe tests\pilot_build_check.py 'D:\Results\Baseline' 'D:\Results\Candidate'
.\.venv\Scripts\python.exe tests\scan_integrity_check.py 'D:\Results\Candidate'
```
