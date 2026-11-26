"""Verify grouped pilot review navigation, persistence, and the guided handoff."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading

from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batch_app import build_app


def check(directory, screenshot):
    # Copy the manifest and representative thumbnails into a disposable review.
    with tempfile.TemporaryDirectory(prefix='pair-review-') as temporary:
        output = Path(temporary)
        source = Path(directory).resolve()
        (output / 'manifest.json').write_bytes((source / 'manifest.json').read_bytes())
        manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        thumbs = output / 'thumbs'
        thumbs.mkdir()
        for group in manifest['groups']:
            thumbnail = f'{group["representative"]}.jpg'
            shutil.copy2(source / 'thumbs' / thumbnail, thumbs / thumbnail)
        server = build_app(output / 'jobs', existing=output)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(channel='chrome', headless=True)
                page = browser.new_page(viewport={'width': 1400, 'height': 900})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{server.server_port}/batches/pilot/')
                visible = page.locator('#compare-cards img')
                page.wait_for_function('document.querySelectorAll("#compare-cards img").length === 2')
                page.wait_for_function('Array.from(document.querySelectorAll("#compare-cards img")).every(img => img.naturalWidth > 1000)')
                for step in ('issues-tab', 'slides-tab', 'export-tab'):
                    assert page.locator('#'+step).is_disabled()
                visible.first.click()
                page.get_by_role('dialog', name='Enlarged slide').wait_for()
                page.keyboard.press('Escape')
                original_count = page.evaluate('pairs().length')
                initial = page.evaluate('groups.length')
                picker = page.locator('#pair-picker')
                page.locator('#next-pair').click()
                expect(picker).to_have_value('1')
                page.locator('#previous-pair').click()
                page.screenshot(path=str(screenshot))
                with page.expect_response(lambda r: r.url.endswith('/decisions') and r.ok):
                    page.locator('#keep-both').click()
                assert page.evaluate('pairs().length') == original_count - 1
                assert page.evaluate('groups.length') == initial
                state = json.loads((output / 'review-state.json').read_text(encoding='utf-8'))
                assert state['resolved_pairs']
                page.reload()
                page.wait_for_function(f"typeof pairs === 'function' && pairs().length === {original_count - 1}")
                with page.expect_response(lambda r: r.url.endswith('/decisions') and r.ok):
                    page.locator('#keep-both').click()
                assert page.evaluate('pairs().length') == original_count - 2
                page.locator('#undo-pair').click()
                assert page.evaluate('pairs().length') == original_count - 1
                assert page.evaluate('groups.length') == initial
                page.set_viewport_size({'width': 390, 'height': 844})
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                assert visible.count() == 2
                while page.locator('#keep-both').is_visible():
                    page.locator('#keep-both').click()
                page.locator('#continue-step').click()
                assert page.locator('#issues-tab').get_attribute('aria-current') == 'step'
                page.locator('#issue-preview img').evaluate('(img) => img.decode()')
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                page.set_viewport_size({'width': 1400, 'height': 900})
                page.screenshot(path=str(screenshot.with_stem(screenshot.stem+'-issues')))
                assert not errors, errors
                browser.close()
        finally:
            server.shutdown()
            server.batches.close()
            server.server_close()
            thread.join()
    print('Grouped pilot review passed with native previews, navigation, keep both, undo, persistence and handoff to issues.')


if __name__ == '__main__':
    check(sys.argv[1], Path(sys.argv[2]).resolve())
