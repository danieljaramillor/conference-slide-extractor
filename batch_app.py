"""Local browser entry point for submitting archives and reviewing batches."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import threading
from urllib.parse import unquote, urlsplit
import webbrowser
import zipfile

from review_server import review_handler

APP_DIR = Path(__file__).resolve().parent
JOB_ID = re.compile(r'[0-9a-f]{16}')
PROGRESS_LINES = ('Processing ', 'Read ', 'Offline OCR:', 'OCR ', 'Writing ')


class Batches:
    def __init__(self, directory, workers, ocr_workers, existing=None):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.ocr_workers = ocr_workers
        self.existing = Path(existing).resolve() if existing else None
        self.existing_summary = self.summary(self.existing) if self.existing else None
        self.lock = threading.Lock()
        self.active = None
        self.process = None
        self.closing = False
        self.reviews = {}

    @staticmethod
    def summary(directory):
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        audit = manifest.get('audit', {})
        summary = {'stats': manifest['stats'], 'ocr_errors': audit.get('ocr_errors', 0)}
        if speaker := audit.get('speaker_detection'):
            summary['speaker_check'] = speaker['status']
            summary['speaker_excluded'] = len(speaker['auto_excluded'])
        if transitions := audit.get('transition_detection'):
            summary['transitions_excluded'] = len(transitions.get('auto_excluded', ()))
            summary['possible_blends'] = len(transitions.get('needs_review', ()))
        return summary

    def write_status(self, job, status):
        temporary = job / 'batch.tmp'
        temporary.write_text(json.dumps(status, ensure_ascii=False), encoding='utf-8')
        temporary.replace(job / 'batch.json')

    def public_status(self, job, status):
        result = job / 'result'
        response = {**status, 'id': job.name}
        if status['state'] in {'uploading', 'running'} and self.active != job.name:
            response.update(state='failed', progress='This batch was interrupted. Submit the ZIP again to retry.')
        if response['state'] == 'ready' and 'stats' not in response:
            response.update(self.summary(result))
        return response

    def list(self):
        jobs = []
        with self.lock:
            for job in self.directory.iterdir():
                if not JOB_ID.fullmatch(job.name) or not (job / 'batch.json').is_file():
                    continue
                status = json.loads((job / 'batch.json').read_text(encoding='utf-8'))
                jobs.append(self.public_status(job, status))
            if self.existing:
                jobs.append({'id': 'pilot', 'name': 'Conference pilot', 'created': '', 'state': 'ready',
                             **self.existing_summary})
            return {'busy': self.active is not None, 'batches': sorted(jobs, key=lambda j: j['created'], reverse=True)}

    def reserve(self, name):
        with self.lock:
            if self.active or self.closing:
                raise RuntimeError('A batch is already running. Submit another ZIP when it finishes.')
            job = self.directory / secrets.token_hex(8)
            job.mkdir()
            status = {'name': name, 'created': datetime.now(timezone.utc).isoformat(),
                      'state': 'uploading', 'progress': 'Receiving ZIP'}
            self.write_status(job, status)
            self.active = job.name
            return job, status

    def finish(self, job, status, state, progress):
        with self.lock:
            try:
                status.update(state=state, progress=progress)
                self.write_status(job, status)
            finally:
                self.active = None
                self.process = None

    def run(self, job, status):
        command = [sys.executable]
        if not getattr(sys, 'frozen', False):
            command.extend(['-u', str(APP_DIR / 'slide_dedupe.py')])
        command.extend(['scan', str(job / 'input.zip'), '--output', str(job / 'result'),
                        '--workers', str(self.workers), '--ocr-workers', str(self.ocr_workers),
                        '--ocr-cache', str(self.directory / '.ocr-cache')])
        lines = []
        try:
            with self.lock:
                if self.closing:
                    raise RuntimeError('The application closed before processing started.')
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding='utf-8', errors='replace')
                self.process = process
                status.update(state='running', progress='Starting extraction')
                self.write_status(job, status)
            with process.stdout as stdout, (job / 'processing.log').open('w', encoding='utf-8') as log:
                for line in stdout:
                    log.write(line)
                    log.flush()
                    line = line.strip()
                    lines = (lines + [line])[-15:]
                    if line.startswith(PROGRESS_LINES):
                        with self.lock:
                            status['progress'] = line
                            self.write_status(job, status)
            if process.wait() or not (job / 'result/manifest.json').is_file():
                message = next((line for line in reversed(lines) if line.startswith('Error:')), '')
                raise RuntimeError(message or 'Processing stopped. See processing.log in the batch folder.')
            status.update(self.summary(job / 'result'))
        except (OSError, ValueError, RuntimeError) as exc:
            self.finish(job, status, 'failed', str(exc))
        else:
            self.finish(job, status, 'ready', 'Extraction complete')

    def review(self, job_id):
        with self.lock:
            if job_id not in self.reviews:
                directory = self.existing if job_id == 'pilot' else self.directory / job_id / 'result'
                if directory is None or not (directory / 'manifest.json').is_file():
                    raise FileNotFoundError('This batch is not ready for review.')
                if job_id != 'pilot':
                    status = json.loads((directory.parent / 'batch.json').read_text(encoding='utf-8'))
                    if status['state'] != 'ready':
                        raise FileNotFoundError('This batch is not ready for review.')
                self.reviews[job_id] = review_handler(directory)
            return self.reviews[job_id][0]

    def close(self):
        with self.lock:
            self.closing = True
            if self.process and self.process.poll() is None:
                self.process.terminate()
            for _, source in self.reviews.values():
                source.close()


def build_app(directory, port=0, workers=8, ocr_workers=4, existing=None, max_upload_gib=4):
    batches = Batches(directory, workers, ocr_workers, existing)
    upload_token = secrets.token_urlsafe(32)
    review_route = re.compile(r'^/batches/(pilot|[0-9a-f]{16})(/.*)?$')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def end_headers(self):
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
            self.send_header('X-Frame-Options', 'DENY')
            super().end_headers()

        def respond(self, body, mime='application/json', status=200):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def local_host(self):
            port = self.server.server_port
            return self.headers.get('Host') in {f'127.0.0.1:{port}', f'localhost:{port}'}

        def mount_review(self, method):
            path = urlsplit(self.path)
            match = review_route.fullmatch(path.path)
            if not match:
                return self.respond({'error': 'Page not found'}, status=404)
            job_id, subpath = match.groups()
            if not subpath and method == 'GET':
                self.send_response(303)
                self.send_header('Location', f'/batches/{job_id}/')
                self.send_header('Content-Length', '0')
                return self.end_headers()
            try:
                handler = batches.review(job_id)
            except (OSError, ValueError) as exc:
                return self.respond({'error': str(exc)}, status=404)
            self.from_dashboard = True
            self.path = (subpath or '/') + ('?' + path.query if path.query else '')
            if method == 'GET':
                return handler.review_get(self)
            return handler.review_post(self)

        def do_GET(self):
            if not self.local_host():
                return self.respond({'error': 'Local application requests only'}, status=403)
            if self.path == '/':
                page = (APP_DIR / 'batch_app.html').read_text(encoding='utf-8')
                style = (APP_DIR / 'app.css').read_text(encoding='utf-8')
                first_run = not (batches.directory / '.quick-start-seen').exists()
                page = page.replace('__FIRST_RUN__', json.dumps(first_run))
                page = page.replace('__APP_STYLE__', style).replace('__UPLOAD_TOKEN__', upload_token)
                return self.respond(page.encode('utf-8'), 'text/html; charset=utf-8')
            if self.path == '/server-guide':
                return self.respond((APP_DIR / 'docs/server-guide.html').read_bytes(), 'text/html; charset=utf-8')
            if self.path == '/practice.zip':
                return self.respond((APP_DIR / 'assets/practice.zip').read_bytes(), 'application/zip')
            if self.path == '/jobs':
                return self.respond(batches.list())
            return self.mount_review('GET')

        def do_POST(self):
            if not self.local_host():
                return self.respond({'error': 'Local application requests only'}, status=403)
            if self.path not in ('/jobs', '/quick-start/seen'):
                return self.mount_review('POST')
            if not secrets.compare_digest(self.headers.get('X-Batch-Token', ''), upload_token):
                return self.respond({'error': 'Reload the batch screen and try again.'}, status=403)
            if self.path == '/quick-start/seen':
                try:
                    (batches.directory / '.quick-start-seen').touch()
                except OSError:
                    return self.respond({'error': 'Could not save your tutorial preference.'}, status=500)
                return self.respond({'saved': True})
            name = unquote(self.headers.get('X-Archive-Name', '')).replace('\\', '/').rsplit('/', 1)[-1]
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not (0 < size <= max_upload_gib * 1024**3) or not name.lower().endswith('.zip'):
                    raise ValueError(f'Choose a ZIP file up to {max_upload_gib} GiB.')
                if shutil.disk_usage(batches.directory).free < size + 512 * 1024**2:
                    raise ValueError('The batch disk needs more free space.')
                job, status = batches.reserve(name[:255])
            except (OSError, ValueError, RuntimeError) as exc:
                return self.respond({'error': str(exc)}, status=400)
            upload = job / 'input.zip'
            try:
                self.connection.settimeout(60)
                with upload.open('wb') as stream:
                    remaining = size
                    while remaining:
                        chunk = self.rfile.read(min(remaining, 1024**2))
                        if not chunk:
                            raise ValueError('ZIP transfer ended early. Select it again to retry.')
                        stream.write(chunk)
                        remaining -= len(chunk)
                if not zipfile.is_zipfile(upload):
                    raise ValueError('The selected file is not a readable ZIP archive.')
            except (OSError, ValueError) as exc:
                upload.unlink(missing_ok=True)
                batches.finish(job, status, 'failed', str(exc))
                return self.respond({'error': str(exc)}, status=400)
            threading.Thread(target=batches.run, args=(job, status), daemon=True).start()
            return self.respond({'id': job.name}, status=202)

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.batches = batches
    return server


def main(argv=None, *, open_browser=False, data_directory=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'Other commands: scan, serve, export, init-server, run-inbox. '
        'Add --help after any command for its options.'))
    parser.add_argument('--data', default=str(data_directory or APP_DIR / 'batches'))
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--ocr-workers', type=int, default=4)
    parser.add_argument('--max-upload-gib', type=int, default=4)
    parser.add_argument('--review', help='Add a previously completed batch to the screen')
    browser = parser.add_mutually_exclusive_group()
    browser.add_argument('--open', dest='open', action='store_true')
    browser.add_argument('--no-browser', dest='open', action='store_false')
    parser.set_defaults(open=open_browser)
    args = parser.parse_args(argv)
    if min(args.workers, args.ocr_workers, args.max_upload_gib) < 1:
        parser.error('Worker counts and maximum ZIP size must be positive.')
    server = build_app(args.data, args.port, args.workers, args.ocr_workers, args.review, args.max_upload_gib)
    url = f'http://127.0.0.1:{server.server_port}/'
    print(f'Open {url} in your browser. Keep this window open while processing.', flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.batches.close()
        server.server_close()


if __name__ == '__main__':
    main()
