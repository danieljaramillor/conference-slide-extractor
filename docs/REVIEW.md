# Review and troubleshoot

## Make decisions in Compare pairs

The queue shows groups the app considered similar enough for a closer comparison. The left and right previews show the currently selected capture from each group. Click either to see its native-resolution crop.

| What you see | What to select |
| --- | --- |
| The same slide twice | **Merge, keep left** or **Merge, keep right**. Prefer the sharper capture with the entire slide visible. |
| One capture adds information while keeping everything in the other | Merge and keep the more complete capture. |
| A chart, column, value, or panel is different and both captures contain unique information | **Keep both slides**. |
| You cannot tell yet | Select **Next** or choose another pair from the dropdown. The skipped pair stays in the queue. |

Pair decisions move you to the next unresolved suggestion and save to the batch. **Undo last pair decision** is available during the current review session. The **Slide issues** tab shows flagged crops, blur, unreadable text, and other issues that need inspection one slide at a time. **All slides** includes the complete batch and the **Excluded** filter. Use the capture list to choose a different image or split a capture out of its group.

If a crop is wrong, select **Adjust crop**, drag around the slide, and apply it. Export refreshes OCR for changed crops and selected captures that lack text. The ZIP includes `slides/`, `originals/`, `text/`, and `mapping.csv`.

## Choose where the ZIP goes

In desktop Chrome or Edge, click **Export ZIP…** to open the browser’s Save As dialog. Choose a folder and filename. The app prepares the export and writes it to that file. Canceling Save As cancels the export. Some embedded browsers only show **Export slides**. In that case use the **Download export ZIP** link and enable the browser’s “Ask where to save each file before downloading” option if you need to choose a folder.

## Common problems

- **Browser did not open.** Open the `http://127.0.0.1:…` address printed in the app window.
- **A batch failed.** Read its status and `processing.log` in the batch folder. Check that the ZIP opens and that the data drive has enough space. If processing was interrupted, submit the ZIP again.
- **Some images were skipped.** Open the batch’s unreadable-images list. Supported images include JPEG, PNG, WebP, BMP, and TIFF.
- **Windows blocks the executable.** Check the SHA-256 hash against `SHA256SUMS.txt` and follow your organization’s software policy. Ask IT if an unsigned executable requires approval.
- **The browser cannot connect to a server.** Use Remote Desktop, or open an SSH tunnel like `ssh -N -L 8765:127.0.0.1:8765 operator@server` from the reviewing computer, then visit `http://127.0.0.1:8765/`. The app listens only on the server’s loopback interface.
