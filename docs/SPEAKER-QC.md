# Speaker-only exclusion QA

The Windows app checks for a face in camera footage using the bundled YuNet model. It also checks the crop, layout, and recognized text before excluding a group. A readable slide or recovered presentation projection stays eligible for export. If an alternate capture in the group preserves the slide, the app can select it. Uncertain captures go to **Slide issues**. Automatic exclusions remain recoverable in **Review excluded captures**. [OpenCV Model Zoo and license](../assets/yunet/README.md)

## Conference dry run

The team visually screened all 1,450 cropped screenshots and reviewed the originals of the eight camera-only candidates. The checks also covered true slides with headshots, a presenter beside a slide, and stage projections. On this pilot the previous release excluded seven of eight speaker-only captures. The new scan excluded all eight and made no other automatic exclusions. The original camera shot it missed showed a presenter looking down at a lectern.

| Full pilot scan | v2.5.0 | v2.6.0 |
| --- | ---: | ---: |
| Captures read | 1,450 | 1,450 |
| Groups | 149 | 149 |
| Suggested pairs | 48 | 48 |
| Speaker-only captures omitted | 7 | 8 |
| Slide candidates exported by the scan | 142 | 141 |
| OCR processing errors | 0 | 0 |

The team checked all 48 suggested pairs for unique content and confirmed the existing grouping keeps the different study-table values and columns separate. The initial scan still includes branding and some transition frames. The pilot’s saved human review already excludes all 12 incomplete or blended transitions found in this pass. For new batches, check those cases during pair review, Slide issues, and the final deck pass.

A stress test exposed a missed face at quarter resolution. After the fix, all eight cropped speaker images were detected at one-eighth resolution with zero false automatic exclusions across 132 transformed slide controls. The test isolated face inference. The full scan also has to locate the crop and read the text on a low-resolution input. One deliberately tilted camera frame still produced no speaker signal. Bright or ambiguous camera frames can require a quality check.

## Verification

- Source regression tests exercise controlled crops, charts with changed numbers, slide builds, real camera and portrait images, and model or OCR failures.
- The Windows build uses a fresh mixed-layout fixture with a speaker-only inset. Its browser tests check the ZIP contents, restored exclusions, merges, splits, and the complete upload-to-export flow.
- On the pilot, the team verified every included original file and every exported PNG against the source, plus every export mapping and OCR sidecar. OCR text accuracy still depends on the capture.
- The packaged browser produced a 125-slide test ZIP using a separate copy of the existing review. It omitted all eight speaker-only captures and the 12 known transitions. Test acknowledgments exercised the workflow, and the saved pilot review stayed in its original batch.

To repeat the file verification on a completed scan from a developer checkout:

```powershell
.\.venv\Scripts\python.exe tests\scan_integrity_check.py 'D:\Results\Session'
```

The command accepts `--export-directory` for a reviewed export folder that includes `review-decisions.json`. Image archives, test exports, and review decisions for the pilot are kept with the local batches.
