"""Use a local face detector and slide evidence to distinguish camera footage."""
from pathlib import Path

import cv2
import numpy as np


MODEL = Path(__file__).with_name('assets') / 'yunet' / 'face_detection_yunet_2023mar.onnx'
ENGINE = 'opencv-yunet-2023mar'
SPEAKER_ONLY = 'probable speaker-only frame (auto-excluded)'
CHECK_SPEAKER = 'check whether slide content is visible'


def create_detector():
    # Passing bytes also works when a Windows username or workspace path contains non-ASCII text.
    return cv2.FaceDetectorYN.create('onnx', MODEL.read_bytes(), b'', (640, 640), .72, .3, 200)


def has_slide_text(item):
    """Keep useful text inside the frame while treating lower thirds as ambiguous."""
    ocr = item.get('ocr', {})
    if ocr.get('status') != 'ok':
        return False
    lines = ocr.get('lines', [])
    if not lines:
        return ocr.get('characters', 0) >= 15
    height = item['slide_height']
    body = []
    for line in lines:
        count = sum(char.isalnum() for char in line['text'])
        if line['score'] < .78 or count < 3:
            continue
        center = sum(point[1] for point in line['box']) / len(line['box']) / height
        if .04 <= center < .78:
            if count >= 10 or (line['score'] >= .9 and line.get('height', 0) / height >= .033):
                return True
            body.append(count)
    return sum(body) >= 15


def is_protected_slide(item):
    return item['crop_mode'] in {'manual', 'projected-screen', 'presenter-slide'} or has_slide_text(item)


def speaker_status(detector, slide, item):
    """Exclude only clear dark camera shots; flag ambiguous images for review."""
    if is_protected_slide(item):
        return None
    height, width = slide.shape[:2]
    # Normalize small camera crops for face inference only. Keep the native export pixels.
    scale = min(4., 640 / max(height, width))
    width, height = max(2, round(width * scale)), max(2, round(height * scale))
    interpolation = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
    image = cv2.resize(slide, (width, height), interpolation=interpolation)
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    if faces is None:
        return None
    people = []
    for face in faces:
        x, y, w, h = face[:4] / [width, height, width, height]
        area = w * h
        if area >= .012 and h >= .13 and .08 < x + w / 2 < .92 and y < .67:
            people.append((area, float(face[-1])))
    if not people:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark_stage = float(np.mean(gray < 70)) >= .76
    confident_face = any(area >= .017 and score >= .80 for area, score in people)
    verified_text = item.get('ocr', {}).get('status') == 'ok'
    camera_inset = item['crop_mode'] in {'auto-inset', 'auto-perspective'}
    if dark_stage and confident_face and verified_text and camera_inset and not item.get('crossfade'):
        return 'speaker-only'
    return 'review'
