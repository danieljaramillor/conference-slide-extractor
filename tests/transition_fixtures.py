"""Generated captures for build preservation and blended-transition tests."""

import cv2
import numpy as np


def slide_build_stages():
    base = np.full((540, 960, 3), 248, np.uint8)
    ink = (100, 45, 10)
    cv2.putText(base, 'Trial results', (40, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.3, ink, 2)
    cv2.putText(base, 'Treatment outcomes: cohort A', (40, 150), cv2.FONT_HERSHEY_SIMPLEX, .8, ink, 2)
    cv2.rectangle(base, (50, 210), (400, 440), (150, 80, 25), 3)
    cv2.line(base, (70, 420), (370, 240), ink, 3)
    for index in range(8):
        cv2.putText(base, f'Cohort {index}: baseline outcome for patients', (50, 192 + index * 36),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (90, 30, 10), 2)

    panel = base.copy()
    cv2.putText(panel, '47.8%', (658, 322), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 2)
    cv2.rectangle(panel, (646, 286), (774, 344), (80, 30, 90), 3)

    complete = panel.copy()
    code = cv2.QRCodeEncoder_create().encode('Slide appendix')
    code = cv2.resize(code, (110, 110), interpolation=cv2.INTER_NEAREST)
    complete[375:485, 700:810] = cv2.cvtColor(code, cv2.COLOR_GRAY2BGR)
    return base, panel, complete


def timestamped(frames):
    return [(f'image_00-00-{4 * index:02}.png', frame) for index, frame in enumerate(frames)]


def build_transition_sequence():
    base, panel, complete = slide_build_stages()
    next_slide = np.full(complete.shape, 70, np.uint8)
    cv2.rectangle(next_slide, (90, 150), (870, 410), (25, 110, 205), -1)
    cv2.putText(next_slide, 'Discussion panel', (56, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 3)
    cv2.putText(next_slide, 'Questions on the published results', (140, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (240, 244, 255), 3)
    blend = cv2.addWeighted(complete, .54, next_slide, .46, 0)
    return timestamped((base, panel, complete, blend, next_slide, next_slide))


def one_sided_sequence():
    """The outgoing disclosure is visible only as faint text in the blend."""
    _, _, incoming = slide_build_stages()
    outgoing = np.full_like(incoming, 248)
    ink = (95, 40, 20)
    cv2.putText(outgoing, 'Required disclosures', (50, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1.27, ink, 2, cv2.LINE_AA)
    disclosures = (
        'Participating centers received support for data collection',
        'Support came from the Independent Methods Consortium',
        'Median follow-up was 31.7 months across both cohorts',
        'Full eligibility criteria are listed in the appendix',
    )
    for index, line in enumerate(disclosures):
        cv2.putText(outgoing, line, (58, 122 + index * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, .72, ink, 2, cv2.LINE_AA)
    blend = cv2.addWeighted(outgoing, .29, incoming, .71, 0)
    unrelated = np.full_like(incoming, (45, 34, 28))
    cv2.putText(unrelated, 'Session overview', (135, 245),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (220, 220, 220), 3, cv2.LINE_AA)
    cv2.putText(unrelated, 'Welcome and agenda', (136, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (190, 195, 205), 2, cv2.LINE_AA)
    return timestamped((unrelated, unrelated, blend, incoming, incoming)), outgoing
