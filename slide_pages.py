"""Recognize printed footer numbers supported by adjacent slide pages."""

import re


_BARE_NUMBER = re.compile(r'[1-9][0-9]{0,2}')


def footer_number(ocr, width, height):
    candidates = []
    for line in ocr.get('lines', []):
        text = line['text'].strip()
        if line['score'] < .96 or not _BARE_NUMBER.fullmatch(text):
            continue
        xs, ys = zip(*line['box'])
        right = max(xs) / width
        top, bottom = min(ys) / height, max(ys) / height
        if (min(xs) / width >= .88 and .92 <= right <= 1 and .92 <= top < bottom <= 1
                and .007 <= bottom - top <= .045):
            candidates.append((int(text), right, (top + bottom) / 2))
    return candidates[0] if len(candidates) == 1 else None


def confirm_footer_numbers(records, candidates):
    """Set observed numbers only in runs with a stable, incrementing footer."""
    def adjacent(first, second):
        return (first is not None and second is not None and second[0] == first[0] + 1
                and abs(first[1] - second[1]) <= .018 and abs(first[2] - second[2]) <= .014)

    for index, triple in enumerate(zip(candidates, candidates[1:], candidates[2:])):
        first, middle, last = triple
        pages = records[index:index+3]
        if (adjacent(first, middle) and adjacent(middle, last)
                and all(page['printed_slide_number'] in (None, observed[0])
                        for page, observed in zip(pages, triple))):
            for page, observed in zip(pages, triple):
                page['printed_slide_number'] = observed[0]
