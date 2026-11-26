# Conference Slide Extractor 2.13.0

- Reads ruled-table cells individually when an OCR box crosses a boundary. `slides.jsonl` records the locations of cells recovered by a reread or left for verification.
- Checks the original pixels when OCR disagrees on citations and author affiliation markers. Checks nearby photographs of projected slides for matching content and keeps the direct capture when the match is clear.
- Keeps both captures when a possible numeric change remains unresolved. **Slide issues** shows the corresponding native image crops and OCR readings side by side.
- Rescan an existing ZIP to apply the updated matching and review checks.

Download `ConferenceSlideExtractor-Windows-x64.exe` under **Assets** and double-click it. See the repository README for storage, Windows Server, and command-line use. Use `SHA256SUMS.txt` to verify the download. The Windows executable is unsigned.
