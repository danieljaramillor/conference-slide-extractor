"""Generated broadcast captures with separately known native slide pixels."""

import cv2
import numpy as np

from mixed_fixtures import make_slide
from speaker_fixtures import load_portrait
from transition_fixtures import slide_build_stages


SLIDE_BOUNDS = (15, 155, 1444, 959)


def broadcast_frame(slide, portrait, presenter_on_right):
    left, top, right, bottom = SLIDE_BOUNDS
    fade = np.linspace((118, 65, 24), (182, 164, 146), 1040, dtype=np.uint8)
    frame = np.full((1080, 1920, 3), (54, 27, 15), dtype=np.uint8)
    frame[20:1060, 8:1908] = fade[:, None, :]
    frame[top:bottom, left:right] = cv2.resize(slide, (right - left, bottom - top))

    frame[155:480, 1461:1900] = (54, 28, 17) if presenter_on_right else (16, 18, 33)
    face_x = 1715 if presenter_on_right else 1498
    frame[178:398, face_x:face_x + 140] = cv2.resize(portrait, (140, 220))
    cv2.rectangle(frame, (1461, 492), (1900, 797), (111, 96, 84), cv2.FILLED)

    light = (240, 246, 255)
    for words, at, scale in (
        ('Autumn Methods Forum', (40, 61), 1.5),
        ('Clinical methods and analysis', (120, 112), 1.0),
        ('Presenter information', (1475, 547), 0.8),
        ('Broadcast Hall North', (620, 1034), 1.6),
    ):
        cv2.putText(frame, words, at, cv2.FONT_HERSHEY_SIMPLEX, scale, light, 2, cv2.LINE_AA)
    return frame


def presenter_capture_sequence():
    portrait = load_portrait()
    early, interim, complete = slide_build_stages()
    other = make_slide(title='Safety monitoring outcomes')
    framed = [broadcast_frame(slide, portrait, alternate) for slide, alternate in (
        (early, False), (early, True), (early, False), (early, True), (interim, False),
        (complete, True), (complete, False), (other, True),
    )]

    # Here the portrait is part of the slide; it must remain in the PNG.
    full_slide = make_slide(title='Working group members')
    full_slide[160:680, 1330:1690] = cv2.resize(portrait, (360, 520))
    cases = [(f'broadcast_{index:02d}.png', frame) for index, frame in enumerate(framed)]
    return [*cases, ('slide_08.png', full_slide)]


def presenter_annotation_sequence():
    """The same chart gains two outlines while the presenter keeps moving."""
    left, top, right, bottom = SLIDE_BOUNDS
    early = make_slide(width=right - left, height=bottom - top, title='Treatment response by cohort')
    cv2.rectangle(early, (780, 150), (1350, 560), (125, 125, 125), 2)
    grid = (95, 95, 95)
    for y in range(185, 560, 48):
        cv2.line(early, (782, y), (1348, y), grid, 2)
    for x in range(810, 1350, 48):
        cv2.line(early, (x, 152), (x, 558), grid, 2)
    for index in range(80):
        x = 800 + 6 * index
        y = 536 - (7 * x + 19 * index * index) % 352
        cv2.circle(early, (x, y), 3, (90, 90, 90), cv2.FILLED)

    orange = (26, 186, 254)
    interim = early.copy()
    cv2.rectangle(interim, (805, 207), (1033, 375), orange, 3)
    complete = interim.copy()
    cv2.rectangle(complete, (1080, 347), (1308, 530), orange, 3)

    portrait = load_portrait()
    return [(f'broadcast_{index:02d}.png', broadcast_frame(slide, portrait, bool(index % 2)))
            for index, slide in enumerate((early, early, early, interim, complete, complete))]
