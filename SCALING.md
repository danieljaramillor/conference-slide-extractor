# Machine sizing and rollout

## Start with a CPU and SSD

The reference machine is a Windows desktop with an Intel Core i7-11700K, 8 physical cores, 16 logical CPUs, and 32 GB RAM. A 516 MB, 1,450-capture pilot took about 4 minutes 40 seconds with an empty OCR cache and about 1 minute 25 seconds with an existing image-text cache. The uncached reference job peaked at about 1.4 GiB of sampled process working set. OCR demand varies with the number of unique slides, text density, and visual changes. Capacity planning should use uncached timings for new conferences.

| Use | Starting specification | Initial settings |
| --- | --- | --- |
| One operator, occasional batches | 4 modern CPU cores, 8 GB RAM, SSD | Start with 4 image workers and 2 OCR workers. Expect slower runs. |
| Recommended workstation | 8 modern CPU cores, 16–32 GB RAM, NVMe SSD | 8 image workers and 4 OCR workers, one batch at a time. |
| First team server | 16 vCPU, 32 GB RAM, NVMe SSD | Try two separate CLI jobs with 4 image workers and 2–4 OCR workers each. Measure both completion times before increasing concurrency. |

These are starting configurations based on the pilot and are not measured guarantees for the other hardware. The model runs on CPU. OCR sessions are configured for one inference thread each and use a bounded number of workers. Adding parallel workers can increase contention, so compare job throughput with per-job latency. [ONNX Runtime CPU threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)

On the reference desktop, two simultaneous full jobs with 4 image workers and 2 OCR workers each finished in 98 seconds total. Both used the existing OCR cache and processed the same input into distinct outputs. The combined sampled working set peaked at 1.26 GiB with OCR models bypassed by cache hits. This confirms parallel CLI execution and warm throughput. Use a separate cold-cache test with two different sessions before deciding how many simultaneous new conferences a server should accept.

Allow roughly **2 GB of free working storage per pilot-sized active batch** as a starting budget. The 516 MB input, about 160 MB of initial slide exports, thumbnails, OCR cache, reviewed output and a downloadable archive can accumulate quickly. Measure real arrivals and retention, then size the SSD. Browser input copies the ZIP onto the extraction machine. Where archives already arrive there, processing the existing path with the CLI avoids an extra copy. A high-resolution or mostly unique batch can use substantially more disk and memory.

## Rollout sequence

1. **Team pilot.** Distribute the same release ZIP to operators, open the reviewed ESMO pilot locally, and run several new conferences from different layouts. Collect corrected groups and crop decisions. Check a sample of slides that had no flags, including numerical tables and charts. Track false merges, missed merges, lost text, and review minutes per batch separately.
2. **Server trial.** Run the package on one private server and open the GUI through SSH. Confirm install, storage permissions, OCR, final output, and backup/restart behavior with real input. Compare one and two simultaneous new conferences. Record images per minute, total turnaround, peak memory and scratch use, and the share needing review. Set concurrency from measured results and team turnaround goals.
3. **Automate input and output.** Keep the extraction CLI as the worker. Trigger it after a ZIP or session folder is complete. Use the upstream session ID and input hash to prevent duplicate processing. Give every attempt a distinct output directory. Store process exit status, a completed manifest, review decisions, and the published export under the session ID. Leave human review in the browser and use the manifest to transmit result paths and status to the next system.
4. **Scale by session.** Add a durable queue and separate worker processes when arrivals exceed one machine. Keep a whole presentation together so returning slides can match. If a single recording is too large, segment at session boundaries and validate any cross-segment reconciliation. Share completed artifacts in controlled storage. Add operator authentication, per-batch access, and a retention policy when moving from a private operator tool to a shared application.
5. **Evaluate more automation.** Build an independently labeled set from several real conference formats. Use it to tune what can be auto-accepted with the team’s agreed tolerance for scientific content loss. Keep review available when the crop is uncertain, a chart panel changes, or text is too small. Profile OCR before considering a different model or hardware accelerator.

The packaged GUI schedules one extraction at a time. Independent CLI processes are the current way to test parallel execution. For future integration, carry over the successful-initial-export marker and the existing ZIP/PNG/text/CSV/JSON formats rather than controlling the GUI with scripts.
