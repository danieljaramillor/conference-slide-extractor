# Team release QC

Version 2.1.0 was tested on Windows x64 with Python 3.13 on the supplied 1,450-image conference ZIP and a controlled mixed-layout ZIP. The review decisions and exports for the supplied conference are separate from the distributable program.

## Conference result

| Check | Result |
| --- | --- |
| Reading and grouping | All 1,450 captures read and assigned exactly once |
| Automated selection | 149 groups, 7 probable speaker-only groups excluded, 142 selected |
| Algorithm review queue | 64 selected groups flagged, with 48 possible match pairs |
| Saved visual QC pass | 126 selected after excluding 16 confirmed transitions, speaker shots, or stage captures with a cleaner slide already available |
| Native export check | All 142 initial PNGs verified pixel-for-pixel against the recorded source extraction and all 142 original files verified byte-for-byte |
| Text export check | 142 initial text files verified present, and the reviewed export contains 126 slides, originals, and text files |
| Mixed layouts | 11 controlled inputs produced the expected four content groups, including a numeric change preserved as a distinct result |
| Pilot pair checks | All 24 development pairs matched the corrected visual labels |

The visual audit inspected a contact sheet of every initial representative and examined specific group members at issue. It revealed that three successive study-table frames each show a different populated column. The previous version treated those as duplicates and could lose two columns. The released matcher checks for substantial content both disappearing and appearing within similar slides. The three frames now have separate groups. The old development label for that case was also corrected.

The visual audit also found a crossfade chosen over a clean capture of the same slide. The release identifies some blends of neighboring timestamped frames and prefers the clean capture where available. The timestamp check uses filenames such as `image_01-09-40.jpg`. Other naming schemes and transitions without matching neighbors can still require review. The saved pilot decisions exclude known examples, including fades that the automatic check missed.

This pass verifies the exported files and inspects the representative images. It does not independently label each source capture, transcribe every slide, or establish a universal accuracy rate. The 24 development pairs were used during tuning. Some distinct chart and table views share titles and remain separate when no capture contains all their data. Review flags deliberately remain on ambiguous groups.

## Package, GUI, and local processing

- Installed the Windows release from an extracted ZIP into a fresh Python environment using the included wheels and `pip --no-index`. The environment passed `pip check`.
- Ran cold OCR from that extracted package with Python socket and DNS calls blocked. It completed with zero Python network attempts. The browser smoke test also blocked everything outside the local app address and made no external requests.
- Verified ZIP submission, invalid-ZIP recovery, progress, batch persistence, native-resolution preview, and download of a valid export ZIP from the extracted package. Existing review checks cover selection, splitting, merging, exclusions, saving decisions, crop editing, and direct browser export.
- All 16 regression tests passed. Windows x64 is the verified target. Pinned Linux x86-64 wheels were found for Python 3.13. A live Linux run remains to be done on the destination server.

## Throughput and memory

| Run | Elapsed | Measured memory |
| --- | ---: | ---: |
| One uncached run of the 1,450 captures, before the final table and crossfade corrections | 280.33 s | 1,423 MiB sampled peak working set |
| Final matcher on the same batch with existing OCR cache | 84.65 s | — |
| Two simultaneous final matcher CLI jobs, using the same populated cache | 97.77 s for both | 1,293 MiB sampled peak combined working set |

The final two concurrent results matched the single final run’s memberships, representatives and review pairs. The cold reference checked 264 images with OCR. The final run targeted 237. Runtime depends on how much new text OCR must read. Memory is process working set sampled every two seconds and excludes the browser and system file cache. Neither cached run measures two new conferences with full OCR running at the same time.

The test machine used an Intel Core i7-11700K with 8 cores and 32 GB RAM. The 516 MB input produced 162.8 MB of initial exports and 42.7 MB of thumbnails. The saved reviewed artifact is about 146 MB. Sizing and the deployment sequence are in `SCALING.md`.
