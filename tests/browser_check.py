"""Smoke-test the generated review UI using an existing Chrome installation."""
import json
from pathlib import Path
import sys

from playwright.sync_api import sync_playwright


def check(report_path):
    report_path = Path(report_path).resolve()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe', headless=True)
        page = browser.new_page(viewport={'width':1440,'height':1000}, accept_downloads=True)
        errors = []
        page.on('pageerror',lambda error: errors.append(str(error)))
        page.goto(report_path.as_uri())
        page.wait_for_function("typeof groups !== 'undefined'")
        page.locator('#slides-tab').click()
        page.locator('#filter').select_option('all')
        page.wait_for_selector('.card')
        count = page.locator('.card').count()
        assert count > 1
        page.locator('[data-zoom]').first.click()
        assert page.locator('#lightbox').is_visible()
        page.keyboard.press('Escape')
        assert not page.locator('#lightbox').is_visible()
        first = page.locator('.card').first
        first.locator('.inspect > summary').click()
        if first.locator('[data-rep]').count() > 1:
            second_id = int(first.locator('[data-rep]').nth(1).get_attribute('data-rep'))
            first.locator('[data-rep]').nth(1).check()
            assert page.evaluate('groups[0].representative') == second_id
            page.locator('.card').first.locator('.inspect').evaluate('(el)=>el.open=true')
            page.locator('.card').first.locator('[data-split]').first.click()
            assert page.locator('.card').count() == count+1
            ids = page.evaluate('[groups[0].id,groups[groups.length-1].id]')
            for gid in ids:
                page.locator(f'[data-select="{gid}"]').check()
            page.locator('#merge').click()
            assert page.locator('.card').count() == count
        excluded_before=page.evaluate('groups.filter(g=>g.excluded).length')
        was_excluded=page.locator('[data-exclude]').first.is_checked()
        page.locator('[data-exclude]').first.check()
        page.locator('#filter').select_option('excluded')
        assert page.locator('.card').count() == excluded_before+int(not was_excluded)
        page.locator('#filter').select_option('all')
        page.get_by_text('Review file', exact=True).click()
        with page.expect_download() as downloaded:
            page.locator('#save').click()
        decision_file = report_path.parent/'browser-test-decisions.json'
        downloaded.value.save_as(decision_file)
        saved = json.loads(decision_file.read_text())
        assert sum(len(g['members']) for g in saved['groups']) == page.evaluate('data.images.length')
        assert any(g.get('excluded') for g in saved['groups'])
        page.locator('#loadfile').set_input_files(decision_file)
        page.wait_for_function("document.getElementById('status').textContent==='Decisions loaded'")
        page.reload()
        page.wait_for_function("typeof groups !== 'undefined' && groups[0].excluded === true")
        page.locator('#slides-tab').click()
        page.locator('#filter').select_option('all')
        page.locator('#search').fill('no-such-file-123456789')
        assert page.locator('.card').count() == 0
        page.locator('#search').fill('')
        page.evaluate('localStorage.clear()')
        page.reload()
        page.wait_for_function("typeof groups !== 'undefined'")
        page.screenshot(path=str(report_path.parent/'review-desktop.png'))
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Mobile horizontal overflow'
        page.screenshot(path=str(report_path.parent/'review-mobile.png'))
        assert not errors, errors
        browser.close()
        print(f'Browser checks passed: {count} groups; zoom, selection, split, merge, exclusion, download/import, search, mobile layout; no JavaScript errors.')


if __name__ == '__main__':
    check(sys.argv[1])
