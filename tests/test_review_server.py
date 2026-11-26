import argparse
import contextlib
import io
import json
from pathlib import Path
import re
import tempfile
import threading
import unittest
import urllib.request
import urllib.error

import cv2
import numpy as np

import slide_dedupe as sd
from mixed_fixtures import make_slide
from review_server import build_server


class ReviewServerTests(unittest.TestCase):
    def test_native_preview_crop_and_one_click_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); src=root/'input'; src.mkdir()
            image=make_slide()
            (src/'1.png').write_bytes(cv2.imencode('.png',image)[1].tobytes())
            (src/'2.png').write_bytes(cv2.imencode('.png',make_slide(title='Another slide'))[1].tobytes())
            args=argparse.Namespace(input=str(src),output=str(root/'out'),workers=1,crop=None,builds='complete',ocr='off')
            with contextlib.redirect_stdout(io.StringIO()): sd.run(args)
            m=json.loads((root/'out/manifest.json').read_text())
            server=build_server(root/'out'); thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
            base=f'http://127.0.0.1:{server.server_port}'
            try:
                page=urllib.request.urlopen(base).read().decode()
                token=json.loads(re.search(r'const REVIEW_TOKEN=("[^"]+")',page)[1])
                native=cv2.imdecode(np.frombuffer(urllib.request.urlopen(base+'/native/0.png').read(),np.uint8),1)
                np.testing.assert_array_equal(native,image)
                crop=cv2.imdecode(np.frombuffer(urllib.request.urlopen(base+'/native/0.png?crop=0.1,0.2,0.9,0.8').read(),np.uint8),1)
                np.testing.assert_array_equal(crop,image[216:864,192:1728])
                decision={'source':m['source'],'manifest_id':m['manifest_id'],'groups':m['groups'],'crops':{'0':[.1,.2,.9,.8]}}
                decision['groups'][1]['excluded']=True
                request=urllib.request.Request(base+'/export',data=json.dumps(decision).encode(),headers={'Content-Type':'application/json'})
                with self.assertRaises(urllib.error.HTTPError) as denied:
                    urllib.request.urlopen(request)
                self.assertEqual(denied.exception.code,403)
                request.add_header('X-Review-Token',token)
                result=json.load(urllib.request.urlopen(request))
                self.assertEqual(result['count'],1)
                self.assertTrue((root/'out'/result['download']).is_file())
                final=cv2.imread(str(Path(result['directory'])/'slides/0001.png'))
                np.testing.assert_array_equal(final,crop)
                decision['crops']['0']=[0,0,float('nan'),1]
                with self.assertRaises(ValueError): sd.validate_review(m,decision)
                decision['crops']['0']=[0,0,.000001,1]
                with self.assertRaisesRegex(ValueError,'at least two pixels'): sd.validate_review(m,decision)
            finally:
                server.shutdown(); server.server_close(); server.slide_source.close(); thread.join()

    def test_changed_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); src=root/'input'; src.mkdir()
            (src/'1.png').write_bytes(cv2.imencode('.png',make_slide())[1].tobytes())
            args=argparse.Namespace(input=str(src),output=str(root/'out'),workers=1,crop=None,builds='complete',ocr='off')
            with contextlib.redirect_stdout(io.StringIO()): sd.run(args)
            m=json.loads((root/'out/manifest.json').read_text())
            (src/'1.png').write_bytes(cv2.imencode('.png',make_slide(title='Changed source'))[1].tobytes())
            source=sd.Source(src)
            try:
                with self.assertRaisesRegex(ValueError,'Source image changed'):
                    sd.export(source,m,root/'export')
            finally: source.close()


if __name__=='__main__': unittest.main()
