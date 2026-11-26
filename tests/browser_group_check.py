"""Exercise grouped build review and verify the downloaded slides at native resolution."""

import argparse
from contextlib import contextmanager, redirect_stdout
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile

import cv2
import numpy as np
from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batch_app import build_app
from mixed_fixtures import make_slide
from slide_dedupe import report, run


def prepare_scan(root):
    """Seed only the candidate graph around real images, crops, and OCR."""
    early = make_slide(title='Treatment follow-up results')
    complete = early.copy()
    changed = make_slide(title='Treatment follow-up results', number='42.8')
    ink = (90, 40, 10)
    for frame, heights in ((complete, (120, 185, 245)), (changed, (245, 185, 120))):
        cv2.putText(frame, 'Week 24 outcomes', (1110, 296), cv2.FONT_HERSHEY_SIMPLEX, 1.5, ink, 3, cv2.LINE_AA)
        cv2.rectangle(frame, (1100, 320), (1770, 700), ink, 5)
        for index, height in enumerate(heights):
            x = 1190 + index * 175
            cv2.rectangle(frame, (x, 651 - height), (x + 78, 651), (155, 65, 110), -1)
    landscape = make_slide(title='Separate landscape evidence')
    portrait = make_slide(width=900, height=1600, title='Portrait evidence')
    captures = (early, complete, changed, landscape, portrait)
    names = ('01-earlier.png', '02-complete.png', '03-replacement.png', '04-landscape.png', '05-portrait.png')
    archive = root / 'captures.zip'
    with zipfile.ZipFile(archive, 'w') as zipped:
        for name, image in zip(names, captures):
            ok, compressed = cv2.imencode('.png', image)
            assert ok
            zipped.writestr(name, compressed.tobytes())

    output = root / 'review'
    args = argparse.Namespace(input=archive, output=output, workers=2, crop=(0.0, 0.0, 1.0, 1.0),
                              builds='all', ocr='on', ocr_workers=2, ocr_cache=root / 'ocr-cache')
    with redirect_stdout(io.StringIO()):
        run(args)
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert [image['name'] for image in manifest['images']] == list(names)
    manifest['groups'] = [
        {'id': index + 1, 'representative': index, 'members': [index], 'builds': 0, 'needs_review': True,
         'review_reasons': ['possible duplicate / build']}
        for index in range(len(captures))
    ]
    manifest['review_pairs'] = [
        {'a': 0, 'b': 1, 'group_a': 1, 'group_b': 2, 'coverage_a': .99, 'coverage_b': .70,
         'title_changed': 0.0, 'mae': .015, 'content_replacement': False},
        {'a': 1, 'b': 2, 'group_a': 2, 'group_b': 3, 'coverage_a': .91, 'coverage_b': .91,
         'title_changed': 0.0, 'mae': .02, 'content_replacement': True},
        {'a': 3, 'b': 4, 'group_a': 4, 'group_b': 5, 'coverage_a': .62, 'coverage_b': .63,
         'title_changed': .05, 'mae': .16, 'content_replacement': False},
    ]
    manifest['stats']['review_pairs'] = len(manifest['review_pairs'])
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    report(output, manifest)
    return output, names, captures


@contextmanager
def serve_fixture(output, executable):
    job_path = output.parent / 'jobs'
    if executable is None:
        server = build_app(job_path, existing=output)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f'http://127.0.0.1:{server.server_port}/batches/pilot/'
        finally:
            server.shutdown()
            server.batches.close()
            server.server_close()
            thread.join()
        return

    process = subprocess.Popen([str(executable.resolve()), '--no-browser', '--data', str(job_path),
                                '--review', str(output)], cwd=output.parent, stdout=subprocess.PIPE,
                               text=True, encoding='utf-8')
    try:
        line = process.stdout.readline()
        assert line.startswith('Open http://127.0.0.1:'), line
        yield line.split()[1] + 'batches/pilot/'
    finally:
        if process.poll() is None:
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True, check=False)
        process.wait(timeout=10)
        process.stdout.close()


def check_export(page, names, captures):
    page.locator('#continue-step').click()
    page.get_by_role('button', name='Continue to deck review').click()
    page.get_by_label('Inspect grouped build').select_option('1')
    preview = page.locator('#completed-build')
    expect(preview.locator('[data-side="selected"] [data-zoom="1"]')).to_be_visible()
    expect(preview.locator('[data-side="alternate"] [data-zoom="0"]')).to_be_visible()
    expect(preview.locator('.change-region').first).to_be_visible()
    page.get_by_role('button', name='Close build preview').click()
    expect(preview).to_be_hidden()
    page.get_by_role('button', name='Continue to export').click()
    page.get_by_role('button', name='Export slides').click()
    download = page.get_by_role('link', name='Download export ZIP')
    expect(download).to_be_visible(timeout=30_000)
    with page.expect_download() as downloaded:
        download.click()
    with zipfile.ZipFile(downloaded.value.path()) as archive:
        assert archive.testzip() is None
        rows = list(csv.DictReader(io.StringIO(archive.read('mapping.csv').decode('utf-8-sig'))))
        mapping = {row['slide']: row['selected_source'] for row in rows}
        assert list(mapping.values()) == [names[index] for index in (1, 2, 3, 4)], mapping
        assert (names[1], names[0]) in {(row['selected_source'], row['member_source']) for row in rows}
        assert len([name for name in archive.namelist() if name.startswith('slides/')]) == len(mapping)
        for order, image_index in enumerate((1, 2, 3, 4), 1):
            encoded = archive.read(f'slides/{order:04}.png')
            pixels = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(pixels, captures[image_index])


def choose(page, base_group, candidate_group, left_image, right_image):
    page.locator('#base-picker').select_option(str(base_group))
    page.locator(f'#family-strip [data-variant="{candidate_group}"]').click()
    cards = page.locator('#compare-cards .compare-side')
    expect(cards.nth(0).locator(f'[data-zoom="{left_image}"]')).to_be_visible()
    expect(cards.nth(1).locator(f'[data-zoom="{right_image}"]')).to_be_visible()


def save_choice(page, selector):
    with page.expect_response(lambda response: response.url.endswith('/decisions') and response.ok):
        page.locator(selector).click()


def comparison(page, base, left, right):
    response = page.request.get(f'{base}comparison?a={left}&b={right}')
    assert response.ok, (response.status, response.text())
    return response.json()


def check_saved_separation(browser, output, executable):
    """An earlier keep-both decision must survive a merge of neighboring groups."""
    copied = output.parent / 'guarded-fixture' / 'review'
    copied.mkdir(parents=True)
    shutil.copytree(output / 'thumbs', copied / 'thumbs')
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    manifest['review_pairs'].append({**manifest['review_pairs'][1], 'a': 0, 'group_a': 1})
    (copied / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    state = {key: manifest[key] for key in ('source', 'manifest_id', 'groups')}
    state.update(version=2, crops={}, revision=1, resolved_pairs=['0:2'])
    state_path = copied / 'review-state.json'
    state_path.write_text(json.dumps(state), encoding='utf-8')
    with serve_fixture(copied, executable) as base:
        page = browser.new_page()
        try:
            page.goto(base)
            choose(page, 1, 2, 0, 1)
            expect(page.locator('#keep-right')).to_be_enabled()
            save_choice(page, '#keep-right')
            state = json.loads(state_path.read_text(encoding='utf-8'))
            assert state['resolved_pairs'] == ['0:2']
            assert state['groups'][0]['members'] == [0, 1]
            choose(page, 1, 3, 1, 2)
            expect(page.locator('#pair-reason')).to_contain_text('previous decision')
            expect(page.locator('#keep-left')).to_be_disabled()
            expect(page.locator('#keep-right')).to_be_disabled()
            expect(page.locator('#keep-both')).to_be_enabled()
        finally:
            page.close()


def check_standalone_report(browser, output):
    page = browser.new_page()
    try:
        page.goto((output / 'review.html').resolve().as_uri())
        expect(page.locator('#family-picker option')).to_have_count(2)
        expect(page.locator('#highlight-toggle')).to_be_disabled()
        expect(page.locator('#highlight-status')).to_contain_text('Open the batch in the app')
        expect(page.locator('#compare-cards .change-region')).to_have_count(0)
    finally:
        page.close()


def check(*, executable=None, browser_channel='chrome', screenshot=None):
    with tempfile.TemporaryDirectory(prefix='review-build-family-', ignore_cleanup_errors=True) as temporary:
        output, names, captures = prepare_scan(Path(temporary))
        with serve_fixture(output, executable) as base, sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel=browser_channel, headless=True)
            page = browser.new_page(viewport={'width': 1440, 'height': 940}, accept_downloads=True)
            errors, external = [], []
            page.on('pageerror', lambda error: errors.append(str(error)))
            origin = base.split('/batches/')[0] + '/'

            def local_only(route):
                if route.request.url.startswith(origin):
                    route.continue_()
                else:
                    external.append(route.request.url)
                    route.abort()

            page.route('**/*', local_only)
            page.add_init_script('window.showSaveFilePicker = undefined;')
            page.goto(base)
            expect(page.locator('#family-picker option')).to_have_count(2)
            page.locator('#family-picker').select_option('0')
            expect(page.locator('#family-strip [data-variant]')).to_have_count(3)
            choose(page, 1, 2, 0, 1)
            evidence = comparison(page, base, 0, 1)
            assert evidence['status'] == 'ok' and evidence['regions']['right'], evidence
            assert all(0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1
                       for x1, y1, x2, y2 in evidence['regions']['right']), evidence
            assert evidence['relationship'] == 'right-complete', evidence

            crop_button = page.get_by_role('button', name='Adjust base crop')
            crop_button.click()
            dialog = page.get_by_role('dialog', name='Adjust slide crop')
            expect(dialog).to_be_visible()
            dialog.get_by_role('button', name='Cancel').click()
            expect(dialog).to_be_hidden()
            expect(crop_button).to_be_focused()

            regions = page.locator('#compare-cards .change-region:visible')
            toggle = page.locator('#highlight-toggle')
            expect(toggle).to_be_enabled()
            if toggle.get_attribute('aria-pressed') != 'true':
                toggle.click()
            expect(regions.first).to_be_visible()
            if screenshot:
                page.screenshot(path=str(screenshot))
            toggle.click()
            expect(regions).to_have_count(0)
            toggle.click()
            expect(regions.first).to_be_visible()
            page.set_viewport_size({'width': 390, 'height': 844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.set_viewport_size({'width': 1440, 'height': 940})

            choose(page, 1, 3, 0, 2)
            for selector in ('#keep-left', '#keep-both', '#keep-right'):
                expect(page.locator(selector)).to_be_disabled()
            choose(page, 2, 3, 1, 2)
            evidence = comparison(page, base, 1, 2)
            assert evidence['numeric_conflict'] and evidence['relationship'] == 'content-replaced', evidence
            cropped = page.request.get(f'{base}comparison?a=1&b=2&crop_a=0,0,1,0.17&crop_b=0,0,1,0.17')
            assert cropped.ok and cropped.json()['numeric_conflict'], cropped.text()
            expect(regions.first).to_be_visible()
            expect(page.locator('#keep-left')).to_be_disabled()
            expect(page.locator('#keep-right')).to_be_disabled()
            expect(page.locator('#keep-both')).to_be_enabled()
            save_choice(page, '#keep-both')
            state_path = output / 'review-state.json'
            state = json.loads(state_path.read_text(encoding='utf-8'))
            assert state['resolved_pairs'] == ['1:2']
            assert len(state['groups']) == 5
            page.reload()
            expect(page.locator('#family-picker option')).to_have_count(2)
            page.locator('#family-picker').select_option('0')
            expect(page.locator('#family-strip [data-variant]')).to_have_count(2)
            choose(page, 1, 2, 0, 1)
            expect(page.locator('#keep-right')).to_be_enabled()
            save_choice(page, '#keep-right')
            state = json.loads(state_path.read_text(encoding='utf-8'))
            assert [group['representative'] for group in state['groups']] == [1, 2, 3, 4]
            expect(page.locator('#family-picker option')).to_have_count(1)
            save_choice(page, '#undo-pair')
            state = json.loads(state_path.read_text(encoding='utf-8'))
            assert [group['representative'] for group in state['groups']] == [0, 1, 2, 3, 4]
            assert state['resolved_pairs'] == ['1:2']

            page.locator('#family-picker').select_option('0')
            choose(page, 1, 2, 0, 1)
            save_choice(page, '#keep-right')
            choose(page, 4, 5, 3, 4)
            uncertain = comparison(page, base, 3, 4)
            assert uncertain['status'] == 'uncertain' and not any(uncertain['regions'].values()), uncertain
            expect(page.locator('#highlight-status')).to_contain_text('unavailable', ignore_case=True)
            expect(regions).to_have_count(0)
            save_choice(page, '#keep-both')
            check_export(page, names, captures)
            assert not errors, errors
            assert not external, external
            check_saved_separation(browser, output, executable)
            check_standalone_report(browser, output)
            browser.close()
    print('Grouped review passed with connected variants, registered highlights, safe decisions, undo, reload and exact native ZIP export.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, help='Run the packaged Windows application.')
    parser.add_argument('--managed-browser', action='store_true', help='Use the Chromium installed by Playwright.')
    parser.add_argument('--screenshot', type=Path, help='Optional highlighted family preview PNG.')
    args = parser.parse_args()
    check(executable=args.executable, browser_channel=None if args.managed_browser else 'chrome',
          screenshot=args.screenshot)
