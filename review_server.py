"""Loopback-only native image previews and one-click reviewed export."""
from __future__ import annotations

from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
import shutil
import threading
import time
from urllib.parse import urlsplit,parse_qs,unquote
import webbrowser
import zipfile

import cv2

from slide_regions import extract


def review_handler(directory):
    from slide_dedupe import Source,decode,export,review_page,validate_review
    root=Path(directory).resolve()
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    images={i['id']:i for i in manifest['images']}
    source=Source(manifest['source'])
    token=secrets.token_urlsafe(32)
    review_html=review_page(manifest)
    review_script=Path(__file__).with_name('review.js').read_bytes()
    export_lock=threading.Lock()
    state_lock=threading.Lock()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,directory=str(root),**kwargs)

        def log_message(self,*args):
            pass

        def end_headers(self):
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Cross-Origin-Resource-Policy','same-origin')
            self.send_header('X-Frame-Options','DENY')
            super().end_headers()

        def respond(self,body,mime='application/json',status=200):
            if not isinstance(body,bytes):
                body=json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.headers.get('Host') not in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'):
                return self.respond({'error':'Local review requests only'},status=403)
            self.review_get()

        def review_get(self):
            parsed=urlsplit(self.path)
            path=unquote(parsed.path)
            try:
                if path in ('/','/review.html'):
                    body=review_html
                    saved=None
                    state=root/'review-state.json'
                    if state.exists():
                        try:
                            saved=json.loads(state.read_text(encoding='utf-8'))
                            validate_review(manifest,saved)
                        except (ValueError,KeyError,TypeError):
                            saved=None
                    encoded=json.dumps(saved).replace('<','\\u003c')
                    body=body.replace('const data=',f'const REVIEW_TOKEN={json.dumps(token)};const SAVED_REVIEW={encoded};\nconst data=',1)
                    if getattr(self, 'from_dashboard', False):
                        body=body.replace('<!-- BATCH_LINK -->', '<a href="/"><span aria-hidden="true">← </span>Back to batches</a>', 1)
                    return self.respond(body.encode('utf-8'),'text/html; charset=utf-8')
                if path.startswith(('/native/','/source/')):
                    image_id=int(Path(path).stem)
                    item=images[image_id]
                    image=decode(source.read(item['name']))
                    if path.startswith('/native/'):
                        crop=parse_qs(parsed.query).get('crop')
                        if crop:
                            override=[float(x) for x in crop[0].split(',')]
                            decision={'source':manifest['source'],'manifest_id':manifest.get('manifest_id'),
                                      'groups':manifest['groups'],'crops':{str(image_id):override}}
                            validate_review(manifest,decision)
                            item={**item,'crop':override,'quad':None}
                        image=extract(image,item)
                    ok,encoded=cv2.imencode('.png',image)
                    if not ok:
                        raise ValueError('Image encoding failed')
                    return self.respond(encoded.tobytes(),'image/png')
                if path=='/review.js':
                    return self.respond(review_script,'text/javascript; charset=utf-8')
                target=(root/path.lstrip('/')).resolve()
                valid_asset = (path.startswith(('/thumbs/', '/selected/'))
                               or (target.parent == root and target.name.startswith('reviewed-') and target.suffix == '.zip'))
                if not valid_asset or not target.is_relative_to(root) or target.is_dir():
                    return self.respond({'error':'Not found'},status=404)
                if not target.is_file():
                    return self.respond({'error':'Not found'},status=404)
                with target.open('rb') as stream:
                    self.send_response(200)
                    self.send_header('Content-Type', mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
                    self.send_header('Content-Length', str(target.stat().st_size))
                    self.end_headers()
                    shutil.copyfileobj(stream, self.wfile)
                return
            except ConnectionError:
                return
            except (ValueError,KeyError,OSError) as exc:
                return self.respond({'error':str(exc)},status=400)

        def do_POST(self):
            if self.headers.get('Host') not in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'):
                return self.respond({'error':'Local review requests only'},status=403)
            self.review_post()

        def review_post(self):
            if self.path not in ('/export','/decisions') or not secrets.compare_digest(self.headers.get('X-Review-Token',''),token):
                return self.respond({'error':'Invalid review session'},status=403)
            try:
                size=int(self.headers.get('Content-Length','0'))
            except ValueError:
                return self.respond({'error':'Invalid request size'},status=400)
            if not 0<size<10_000_000:
                return self.respond({'error':'Invalid request size'},status=400)
            try:
                decision=json.loads(self.rfile.read(size))
                validate_review(manifest,decision)
                if self.path=='/decisions':
                    with state_lock:
                        state=root/'review-state.json'
                        try:
                            old=json.loads(state.read_text(encoding='utf-8')) if state.exists() else {}
                        except ValueError:
                            old={}
                        if decision.get('revision',0)>=old.get('revision',0):
                            temp=state.with_suffix('.tmp')
                            temp.write_text(json.dumps(decision,indent=2),encoding='utf-8')
                            temp.replace(state)
                    return self.respond({'saved':True})
                if not export_lock.acquire(blocking=False):
                    return self.respond({'error':'An export is already running'},status=409)
                try:
                    name='reviewed-'+time.strftime('%Y%m%d-%H%M%S')+'-'+secrets.token_hex(3)
                    dest=root/name
                    export(source,manifest,dest,decision['groups'],decision.get('crops'))
                    (dest/'review-decisions.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
                    archive=root/f'{name}.zip'
                    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
                        for file in dest.rglob('*'):
                            if file.is_file():
                                z.write(file,file.relative_to(dest).as_posix())
                    count=sum(not g.get('excluded') for g in decision['groups'])
                    return self.respond({'count':count,'download':archive.name,'directory':str(dest)})
                finally:
                    export_lock.release()
            except Exception as exc:
                return self.respond({'error':str(exc)},status=400)

    return Handler, source


def build_server(directory,port=0):
    handler, source = review_handler(directory)
    server=ThreadingHTTPServer(('127.0.0.1',port),handler)
    server.slide_source=source
    return server


def serve(directory,port=0,open_browser=False):
    server=build_server(directory,port)
    url=f'http://127.0.0.1:{server.server_port}/'
    print(f'Native-resolution review: {url}',flush=True)
    print('Keep this window open while reviewing. Ctrl+C stops the local server.',flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.slide_source.close()
