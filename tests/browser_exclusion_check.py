"""Check default omission, deliberate recovery, and safe edits around speaker exclusions."""

import csv
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import zipfile

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review_server import build_server


SPEAKER_REASON = 'probable speaker-only frame (auto-excluded)'


def prepare_scan(source, target):
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    manifest['groups'] = [group for group in manifest['groups'] if not group.get('auto_exclusion')]
    capture_ids = {member for group in manifest['groups'] for member in group['members']}
    manifest['images'] = [image for image in manifest['images'] if image['id'] in capture_ids]
    for group in manifest['groups']:
        group.update(excluded=False, needs_review=False, review_reasons=[])
    manifest['review_pairs'] = []
    automatic = next(group for group in manifest['groups'] if len(group['members']) > 1)
    legacy, included, *_ = [group for group in manifest['groups'] if group is not automatic]
    automatic.update(excluded=True, needs_review=True, review_reasons=[SPEAKER_REASON], auto_exclusion='speaker-only')
    legacy.update(excluded=True, needs_review=True, review_reasons=[SPEAKER_REASON])
    included.update(needs_review=True, review_reasons=['check whether slide content is visible'])
    images = {image['id']: image for image in manifest['images']}
    high_score = images[automatic['representative']].setdefault('ocr', {})
    high_score['characters'] = 100_000
    (target / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    shutil.copytree(source / 'thumbs', target / 'thumbs')
    return automatic, legacy, included, images, len(manifest['groups']) - 2


def check(directory, *, browser_channel='chrome'):
    with tempfile.TemporaryDirectory(prefix='review-exclusions-', ignore_cleanup_errors=True) as temporary:
        root = Path(temporary)
        automatic, legacy, included, images, initial_count = prepare_scan(Path(directory), root)
        server = build_server(root)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f'http://127.0.0.1:{server.server_port}/'
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(channel=browser_channel, headless=True)
                page = browser.new_page(accept_downloads=True, viewport={'width': 1280, 'height': 900})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.add_init_script('window.showSaveFilePicker = undefined;')
                page.goto(base)

                cards = page.locator('#grid .card')
                excluded_switch = page.locator('#show-excluded')
                continue_review = page.locator('#continue-step')
                assert page.locator('#issues-tab').get_attribute('aria-current') == 'step'
                assert 'Confirm this frame contains a slide' in page.locator('#issue-reasons').inner_text()
                with page.expect_response(lambda response: response.url.endswith('/decisions') and response.ok):
                    page.locator('#issue-check').click()
                continue_review.click()
                assert cards.count() == initial_count
                assert excluded_switch.inner_text() == 'Review excluded captures (2)'
                continue_review.click()
                page.locator('#export').click()
                with page.expect_download() as downloaded:
                    page.get_by_role('link', name='Download export ZIP').click()
                with zipfile.ZipFile(downloaded.value.path()) as archive:
                    mapping = list(csv.DictReader(io.StringIO(archive.read('mapping.csv').decode('utf-8-sig'))))
                    assert len({row['slide'] for row in mapping}) == initial_count
                    selected_names = {row['selected_source'] for row in mapping}
                    assert selected_names.isdisjoint(images[group['representative']]['name'] for group in (automatic, legacy))
                    assert images[included['representative']]['name'] in selected_names
                    assert len([name for name in archive.namelist() if name.startswith('slides/')]) == initial_count

                page.locator('#back-step').click()
                excluded_switch.click()
                assert cards.count() == 2
                assert page.get_by_text('Likely speaker-only', exact=True).count() == 2

                auto_card = page.locator(f'#group-{automatic["id"]}')
                auto_card.locator('.inspect > summary').click()
                split_id = next(member for member in automatic['members'] if member != automatic['representative'])
                with page.expect_response(lambda response: response.url.endswith('/decisions') and response.ok):
                    auto_card.locator(f'[data-split="{split_id}"]').click()
                assert cards.count() == 3
                split_card = cards.last
                assert split_card.get_by_role('button', name='Restore to deck').is_visible()

                legacy_card = page.locator(f'#group-{legacy["id"]}')
                with page.expect_response(lambda response: response.url.endswith('/decisions') and response.ok):
                    legacy_card.get_by_role('button', name='Restore to deck').click()
                assert continue_review.inner_text() == 'Return to slide issues'
                assert page.locator('#export-tab').is_disabled()
                page.reload()
                assert page.locator('#issues-tab').get_attribute('aria-current') == 'step'
                assert 'likely speaker footage' in page.locator('#issue-reasons').inner_text()
                assert page.locator('#issue-preview').get_attribute('data-zoom') == str(legacy['representative'])
                with page.expect_response(lambda response: response.url.endswith('/decisions') and response.ok):
                    page.locator('#issue-check').click()
                continue_review.click()
                assert cards.count() == initial_count + 1

                page.locator('#filter').select_option('all')
                for group in (automatic, included):
                    page.locator(f'#grid [data-select="{group["id"]}"]').check()
                with page.expect_response(lambda response: response.url.endswith('/decisions') and response.ok):
                    page.locator('#merge').click()
                decisions = json.loads((root / 'review-state.json').read_text(encoding='utf-8'))
                mixed = next(group for group in decisions['groups'] if automatic['representative'] in group['members'])
                assert mixed['representative'] == included['representative']
                assert not mixed.get('excluded')
                assert mixed.get('auto_exclusion') is None
                assert page.locator('#export-tab').is_disabled()

                page.locator('#filter').select_option('included')
                while cards.count():
                    with page.expect_response(lambda response: response.url.endswith('/decisions') and response.ok):
                        cards.first.locator('[data-exclude]').click()
                assert continue_review.is_disabled()
                excluded_switch.click()
                assert cards.count() == initial_count + 2
                assert cards.first.get_by_role('button', name='Restore to deck').is_visible()
                assert not errors, errors
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
            server.slide_source.close()
            thread.join()
    print('Speaker exclusion workflow passed: ambiguous review, clean ZIP, recovery, persistence, split, and mixed merge.')


if __name__ == '__main__':
    check(sys.argv[1])
