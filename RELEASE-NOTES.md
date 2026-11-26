# Conference Slide Extractor 2.12.0

- Checks cursor shapes and citation punctuation against the original pixels before splitting slides over a disputed number.
- Keeps a readable diagram state separately when a later callout covers part of it, including when a large panel appears at the same time.
- Recovers more slide crops in recurring broadcast layouts when the same borders and surrounding graphics are visible.
- Writes `slides.jsonl` for each session export. The file includes the export slide number, printed slide number when explicitly recognized, full OCR text, and table rows when the layout supports reconstruction.
- Rescan an existing ZIP to apply the updated matching. Export a reviewed batch again to generate its session text file.

Download `ConferenceSlideExtractor-Windows-x64.exe` under **Assets** and double-click it. See the repository README for storage, Windows Server, and command-line use. Use `SHA256SUMS.txt` to verify the download. The Windows executable is unsigned.
