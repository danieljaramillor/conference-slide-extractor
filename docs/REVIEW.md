# Review and troubleshoot

## Review related captures together

**Review builds** gathers related suggestions into groups so you can see nearby variants in one place. Choose a **Base capture**, then pick a candidate from the capture strip or the suggested-comparison menu. **Highlight changes** marks additions and other differences when the captures align reliably. The app may recommend a more complete capture when the visual evidence supports it. Click either preview to inspect its native-resolution image.

Decisions apply to the two captures in the current suggestion. You can also inspect other combinations within the same group. Return to a suggested comparison to save a decision. If a changed value or distinct content is detected, **Keep both slides** is the available decision.

The app crops out separate presenter panes when it can confirm the slide boundary across neighboring screenshots. A presenter overlay that covers the slide or an ambiguous layout can still need review. If a webcam or broadcast frame remains in the preview, use **Adjust crop** above each image to select the slide pane. The app compares the updated crops and checks the retained crop in **Slide issues** before export.

| What you see | What to select |
| --- | --- |
| The same slide twice | **Merge, keep base** or **Merge, keep candidate**. Prefer the sharper capture with the entire slide visible. |
| One capture adds information while keeping everything in the other | Merge and keep the more complete capture. |
| A chart, column, value, or panel is different and both captures contain unique information | **Keep both slides**. |
| You cannot tell yet | Use the next-comparison arrow or choose another suggestion from the menu. The skipped pair stays in the queue. |

Pair decisions move you to the next unresolved suggestion and save to the batch. **Undo last pair decision** is available during the current review session. Once the pair queue is clear, continue to **Slide issues**.

## Check one flagged slide at a time

In **Slide issues**, read **What to check** next to the current image. Click the image to enlarge it. Use **Adjust crop** to draw a better slide region, or **Choose another capture** if the group has a better screenshot. Use **Mark checked and next** to approve the current capture and move on. **Exclude slide** omits it from the ZIP. **Undo last issue decision** restores the last slide you marked or excluded in this session.

**Check possible changed value** shows the same region from two captures with the original and local OCR readings when available. Compare the pixels and use **Mark checked and next** to keep both slides. If you confirm the two slides are equivalent, select both in **Review deck** and use **Merge selected**.

After applying a crop, check the updated preview before approving it. Export refreshes OCR for changed crops and selected captures that lack text. Slides with small text may receive an additional higher-resolution pass. The ZIP includes `slides/`, `originals/`, `text/`, `mapping.csv`, and `slides.jsonl`. The session text file follows the reviewed deck order and includes text, printed slide numbers supported by a label or a consistent footer sequence, and reconstructed table rows when supported by the image.

## Review the deck

Open **Review deck** when both queues are clear. The default view shows included slides in export order. You can search filenames, inspect alternate captures, split or merge groups, and restore captures through the **Excluded** filter. If an edit creates a new pair or slide issue, complete that check before exporting. Select **Continue to export** when the deck is ready.

Use **Inspect grouped build** to open a build and compare the selected export capture with any alternate. You can toggle highlights and close the preview while keeping the current selection.

Choose **Review excluded captures** to inspect blended transitions, likely speaker-only captures, and manual exclusions. A screenshot with a readable slide can be restored with **Restore to deck**. A restored automatic exclusion returns to **Slide issues** so you can verify the chosen capture before export.

The transition check uses nearby captures to distinguish a short overlap between two slides from a legitimate build. Keep capture order in the filenames. The detector accepts timestamps such as `image_00-10-24.png` and consecutive names such as `frame_102.png`.

If the preceding screenshot is missing, the app looks for a stable following slide and evidence that part of the earlier slide is still visible. When the earlier content is unique or the evidence is uncertain, **Slide issues** keeps the blended frame and offers **View following slide** for comparison. **Mark checked and next** keeps the source capture and its OCR. **Exclude slide** omits it when you decide the overlapping content can be left out.

## Choose where the ZIP goes

In desktop Chrome or Edge, click **Export ZIP…** to open the browser’s Save As dialog. Choose a folder and filename. The app prepares the export and writes it to that file. Canceling Save As cancels the export. Some embedded browsers only show **Export slides**. In that case use the **Download export ZIP** link and enable the browser’s “Ask where to save each file before downloading” option if you need to choose a folder.

## Common problems

- **Browser did not open.** Open the `http://127.0.0.1:…` address printed in the app window.
- **A batch failed.** Read its status and `processing.log` in the batch folder. Check that the ZIP opens and that the data drive has enough space. If processing was interrupted, submit the ZIP again.
- **Some images were skipped.** Open the batch’s unreadable-images list. Supported images include JPEG, PNG, WebP, BMP, and TIFF.
- **Windows blocks the executable.** Check the SHA-256 hash against `SHA256SUMS.txt` and follow your organization’s software policy. Ask IT if an unsigned executable requires approval.
- **The browser cannot connect to a server.** Use Remote Desktop, or open an SSH tunnel like `ssh -N -L 8765:127.0.0.1:8765 operator@server` from the reviewing computer, then visit `http://127.0.0.1:8765/`. The app listens only on the server’s loopback interface.
