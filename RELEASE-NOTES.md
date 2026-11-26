# Conference Slide Extractor 2.8.0

- Detects likely blends when only the following slide is captured cleanly. **Slide issues** points to that capture for comparison.
- Preserves the original frame and OCR when the earlier content may be unique. An independently verified clean capture elsewhere in the sequence can allow automatic omission.
- Keeps preserved blends out of slide-build matching so their extra OCR text cannot displace a complete clean slide.

Download `ConferenceSlideExtractor-Windows-x64.exe` under **Assets** and double-click it. See the repository README for storage, Windows Server, and command-line use. Use `SHA256SUMS.txt` to verify the download. The Windows executable is unsigned.
