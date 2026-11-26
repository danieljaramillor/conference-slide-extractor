import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import cv2
import numpy as np

import slide_dedupe as sd
from slide_regions import detect_region,extract
from slide_ocr import text_evidence
from mixed_fixtures import fixtures


class LayoutTests(unittest.TestCase):
    def test_region_location_and_native_size_across_layouts(self):
        for name,image,expected in fixtures():
            with self.subTest(layout=name):
                region=detect_region(image)
                result=extract(image,region)
                if expected:
                    self.assertLess(max(abs(x-y) for x,y in zip(region['crop'],expected)),.015,region)
                    h,w=image.shape[:2]
                    self.assertEqual(result.shape[1],round(region['crop'][2]*w)-round(region['crop'][0]*w))
                else:
                    self.assertEqual(region['crop_mode'],'auto-perspective',region)
                    self.assertGreater(result.shape[1],1500)
                    self.assertGreater(result.shape[0],900)

    def test_full_slide_png_preserves_pixels_exactly(self):
        image=fixtures()[0][1]
        region=detect_region(image)
        exported=extract(image,region)
        roundtrip=cv2.imdecode(cv2.imencode('.png',exported)[1],1)
        np.testing.assert_array_equal(roundtrip,image)

    def test_weak_artwork_candidate_keeps_full_frame(self):
        # A large rectangular illustration inside a light slide is not enough
        # evidence to discard the title and content outside it.
        image=np.full((1080,1920,3),250,np.uint8)
        cv2.rectangle(image,(0,700),(1919,1079),(170,200,210),-1)
        cv2.putText(image,'Industry participation',(70,140),cv2.FONT_HERSHEY_SIMPLEX,2,(30,30,30),3)
        cv2.putText(image,'Important text outside the illustration',(70,250),cv2.FONT_HERSHEY_SIMPLEX,1,(20,20,20),2)
        cv2.rectangle(image,(900,420),(1780,990),(20,20,20),8)
        for j in range(5):cv2.putText(image,'Website text',(960,500+j*70),cv2.FONT_HERSHEY_SIMPLEX,.8,(30,30,30),2)
        region=detect_region(image)
        self.assertEqual(region['crop_mode'],'full')
        np.testing.assert_array_equal(extract(image,region),image)

    def test_high_resolution_duplicate_wins_over_small_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); archive=root/'pool.zip'
            fs={name:(im,box) for name,im,box in fixtures()}
            with zipfile.ZipFile(archive,'w') as z:
                for name in ['small','wide-full','right','four-three-full','four-three-right','portrait-full','portrait-inset']:
                    z.writestr(name+'.png',cv2.imencode('.png',fs[name][0])[1].tobytes())
            args=argparse.Namespace(input=str(archive),output=str(root/'out'),workers=2,crop=None,builds='complete',ocr='off')
            with contextlib.redirect_stdout(io.StringIO()):
                sd.run(args)
            m=json.loads((root/'out/manifest.json').read_text())
            self.assertEqual(len(m['groups']),3,[(g['members'],g['representative']) for g in m['groups']])
            group=next(g for g in m['groups'] if any(m['images'][i]['name']=='small.png' for i in g['members']))
            self.assertEqual(m['images'][group['representative']]['name'],'wide-full.png')
            seq=m['groups'].index(group)+1
            png=cv2.imread(str(root/'out/selected/slides'/f'{seq:04d}.png'))
            self.assertEqual(png.shape[:2],(1080,1920))

    def test_ocr_numeric_change_blocks_merge(self):
        def item(text):
            return {'slide_width':1920,'slide_height':1080,'ocr':{'lines':[
                {'text':'Trial primary outcome','score':.99,'box':[[40,30],[600,30],[600,70],[40,70]]},
                {'text':text,'score':.99,'box':[[40,200],[900,200],[900,240],[40,240]]}]}}
        self.assertTrue(text_evidence(item('Result: 47.8 percent'),item('Result: 42.8 percent'))['numeric_conflict'])
        self.assertFalse(text_evidence(item('Result: 47.8 percent'),item('Result: 47.8 percent'))['numeric_conflict'])
        self.assertFalse(text_evidence(item('13:30 - 14:30 - YO MENTORSHIP SESSION'),item('13:30 - 14:30 - Y0 MENTORSHIP SESSION'))['numeric_conflict'])
        self.assertFalse(text_evidence(item('oestrogen deprivation by Als2\u00b2-4'),item('oestrogen deprivation by Als2-4'))['numeric_conflict'])


if __name__=='__main__':
    unittest.main()
