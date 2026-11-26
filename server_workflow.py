"""Set up and run a local folder of conference screenshot jobs."""
import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys
import tempfile
from uuid import uuid4


APP_DIR = Path(__file__).resolve().parent


@contextmanager
def runner_lock(root):
    # The OS releases the advisory lock when a scheduled process stops or crashes.
    with (root / '.runner.lock').open('a+b') as lock:
        lock.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Could not lock the server folder. Check for another runner and verify write access.') from exc
        yield


def initialize(root):
    for name in ('inbox', 'runs'):
        (root / name).mkdir(parents=True, exist_ok=True)
    for source in (APP_DIR / 'docs/server-guide.html', APP_DIR / 'assets/practice.zip'):
        target = root / source.name
        if not target.exists():
            with tempfile.NamedTemporaryFile(dir=root, delete=False) as tmp:
                pending = Path(tmp.name)
            try:
                shutil.copyfile(source, pending)
                pending.replace(target)
            finally:
                pending.unlink(missing_ok=True)


def process_job(root, job, archive):
    import slide_dedupe

    result = job / 'result'
    # The scan publishes its manifest last. Keep an interrupted attempt for inspection.
    if result.exists():
        result.rename(job / ('interrupted-' + uuid4().hex[:12]))
    print(f'Processing {archive.name} in {job}', flush=True)
    with (job / 'processing.log').open('a', encoding='utf-8', buffering=1) as log:
        print(datetime.now(timezone.utc).isoformat(), file=log)
        try:
            with redirect_stdout(log), redirect_stderr(log):
                slide_dedupe.main(['scan', str(archive), '--output', str(result),
                                   '--ocr-cache', str(root / 'ocr-cache')])
        except Exception as exc:
            print(f'Error: {exc}', file=log)
            (job / 'failed.txt').write_text(str(exc), encoding='utf-8')
            print(f'Failed: {archive.name}. See {job / "processing.log"}', file=sys.stderr, flush=True)
            return False
    print(f'Ready for review: {result}', flush=True)
    return True


def run_inbox(root):
    runs = root / 'runs'
    jobs = []
    for job in sorted(runs.iterdir()):
        if (job / 'result/manifest.json').is_file() or (job / 'failed.txt').is_file():
            continue
        jobs.extend((job, archive) for archive in (job / 'input').glob('*') if archive.is_file())
    for archive in sorted((root / 'inbox').iterdir()):
        if not archive.is_file() or archive.suffix.lower() != '.zip':
            continue
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        job = Path(tempfile.mkdtemp(prefix=stamp + '-', dir=runs))
        (job / 'input').mkdir()
        saved = job / 'input' / archive.name
        archive.rename(saved)
        jobs.append((job, saved))
    failures = 0
    for job, archive in jobs:
        if not process_job(root, job, archive):
            failures += 1
    print(f'Processed {len(jobs)} batch(es); {failures} failed.', flush=True)
    return 1 if failures else 0


def main(argv):
    parser = argparse.ArgumentParser(prog=f'{Path(sys.argv[0]).name} {argv[0]}',
                                     description='Prepare a server folder.' if argv[0] == 'init-server'
                                     else 'Process all finished ZIPs in the inbox once, then exit.')
    parser.add_argument('folder', type=Path, help='Writable local server workspace')
    root = parser.parse_args(argv[1:]).folder.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with runner_lock(root):
        if argv[0] == 'init-server':
            initialize(root)
            print(f'Server folder ready: {root}\nOpen {root / "server-guide.html"} for the practice run and scheduler setup.')
            return 0
        if not (root / 'inbox').is_dir() or not (root / 'runs').is_dir():
            parser.error('Initialize this folder with init-server first.')
        return run_inbox(root)
