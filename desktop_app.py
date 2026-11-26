"""Windows executable entry point for the browser UI and extraction commands."""
import os
from pathlib import Path
import sys
import zipfile


def main():
    if sys.stdout:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    if sys.stderr:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    args = sys.argv[1:]
    local_app_data = Path(os.environ.get('LOCALAPPDATA') or Path.home() / 'AppData' / 'Local')
    if args and args[0] in {'scan', 'serve', 'export'}:
        import slide_dedupe

        if args[0] == 'scan' and not any(arg == '--ocr-cache' or arg.startswith('--ocr-cache=') for arg in args):
            args += ['--ocr-cache', str(local_app_data / 'Conference Slide Extractor' / 'ocr-cache')]
        slide_dedupe.main(args)
    else:
        from batch_app import main as run_browser_app

        run_browser_app(args, open_browser=True, data_directory=local_app_data / 'Conference Slide Extractor' / 'batches')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f'Error: {exc}', file=sys.stderr, flush=True)
        if len(sys.argv) == 1 and sys.stdin and sys.stdin.isatty():
            input('Press Enter to exit.')
        raise SystemExit(1)
