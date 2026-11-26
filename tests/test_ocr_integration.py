"""End-to-end numeric regression, using bundled offline models rather than mocks."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import cv2

import slide_dedupe as sd
from mixed_fixtures import make_slide,embed


class OCRIntegrationTests(unittest.TestCase):
    def test_small_number_change_is_not_lost_across_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); archive=root/'pool.zip'
            original=make_slide(number='47.8')
            changed=make_slide(number='42.8')
            images=[('1-browser.png',embed(original,(2300,1500),(250,200,1800,1012),(65,65,65))),
                    ('2-changed.png',changed),('3-right.png',embed(original,(2560,1440),(800,180,1600,900))),
                    ('4-native.png',original)]
            with zipfile.ZipFile(archive,'w') as z:
                for name,image in images: z.writestr(name,cv2.imencode('.png',image)[1].tobytes())
            args=argparse.Namespace(input=str(archive),output=str(root/'out'),workers=2,crop=None,
                                    builds='complete',ocr='on',ocr_workers=1,ocr_cache=str(root/'cache'))
            with contextlib.redirect_stdout(io.StringIO()): sd.run(args)
            m=json.loads((root/'out/manifest.json').read_text())
            owner={i:g['id'] for g in m['groups'] for i in g['members']}
            self.assertEqual(owner[0],owner[2])
            self.assertEqual(owner[0],owner[3])
            self.assertNotEqual(owner[0],owner[1])
            self.assertEqual(len(m['groups']),2)
            self.assertTrue(m['audit']['numeric_splits'])
            group=next(g for g in m['groups'] if 0 in g['members'])
            self.assertEqual(group['representative'],3)


if __name__=='__main__': unittest.main()
