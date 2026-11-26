# Reveal and capture quality control

This pass used the original 1,450-image pilot ZIP. It reviewed all 149 representative previews and 48 suggested match pairs. It also visually compared 38 sampled types of changed matches drawn from 84 assignments with a detectable image change. Native images and OCR line positions were checked for the cases below. This was a targeted review of the highest-risk cases. The underlying OCR transcript has not been checked character by character.

| Case | Before | After |
| --- | --- | --- |
| Small result panel revealed | Frame 274 was selected. Its lower-right result panel was still blank. | Frame 278 retains the rest of the slide and includes the panel and its OCR text. |
| Late QR reveal | Frame 1373 was selected before the QR code and its labels appeared. | Frame 1374 includes the added content. |
| Different charts, table columns, and guidance callouts replace one another | The layout and headline can look identical even as the figures change. | Verified 26 stages across eight families remain separate. Keep both if either contains information absent from the other. |
| Slide projected above a dark stage | Frames 716 and 1103 were cropped to the whole stage camera inset. | The detector rectifies the projected slide itself. Recovered images measure 727 × 408 and 730 × 402. Both remain flagged for quality and each has a cleaner direct capture elsewhere in this pilot. |

The first two errors came from a small area of new content passing the visual duplicate threshold. If the later capture keeps nearly all earlier visual detail and adds a localized block, completeness now outranks tie-breaking on image quality. The comparison still checks for replaced content and confidently read numeric changes. The projection fallback requires a dark camera inset around a bright text-bearing quadrilateral with perspective evidence. Other camera layouts may still need manual cropping.

The rerun preserved the partition of all 1,450 inputs into 149 groups. It selected 142 candidate images after automatic speaker-only exclusions. Two representatives and two crop regions changed. The original review remains separate from the rerun. The QC export applies the earlier 16 confirmed exclusions for transitions and redundant camera views, yielding 126 selected images.

## Capture limits

Of the 1,400 timestamped captures, 1,384 consecutive intervals were four seconds, 14 were eight seconds, and one was 16 seconds. A short-lived reveal may fall between frames. If the upstream workflow can capture on slide change, do that. Otherwise, trial a one- or two-second interval on sessions with fast builds, then measure the added processing load. OCR can read available pixels and help distinguish stages. It cannot recover a state absent from the original captures.
