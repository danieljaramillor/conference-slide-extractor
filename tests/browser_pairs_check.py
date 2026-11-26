"""Verify that the pair actions save, keep the chosen capture, and undo safely."""
import json
from pathlib import Path
import sys
import tempfile
import threading

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batch_app import build_app


def check(directory, screenshot):
    # Copy only the metadata. This leaves the pilot review and decisions untouched.
    with tempfile.TemporaryDirectory(prefix='pair-review-') as temporary:
        output = Path(temporary)
        source = Path(directory).resolve()
        (output / 'manifest.json').write_bytes((source / 'manifest.json').read_bytes())
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
                visible.first.click()
                page.get_by_role('dialog', name='Enlarged slide').wait_for()
                page.keyboard.press('Escape')
                original_count = page.evaluate('pairs().length')
                initial = page.evaluate('groups.length')
                page.locator('#next-pair').click()
                assert page.locator('#pair-picker').input_value() == '1'
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
                expected = page.evaluate('pairs()[0].right.representative')
                with page.expect_response(lambda r: r.url.endswith('/decisions') and r.ok):
                    page.locator('#keep-right').click()
                assert page.evaluate('groups.length') == initial - 1
                assert page.evaluate(f'owner({expected}).representative') == expected
                page.locator('#undo-pair').click()
                assert page.evaluate('groups.length') == initial
                page.set_viewport_size({'width': 390, 'height': 844})
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                assert visible.count() == 2
                assert not errors, errors
                browser.close()
        finally:
            server.shutdown()
            server.batches.close()
            server.server_close()
            thread.join()
    print('Pair queue passed: native previews, navigation, keep both, keep right, undo, persistence, and mobile layout.')


if __name__ == '__main__':
    check(sys.argv[1], Path(sys.argv[2]).resolve())
