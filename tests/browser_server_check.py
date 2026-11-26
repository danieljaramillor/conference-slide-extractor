"""Native preview, crop editing, durable state, and browser-triggered export."""
import json
from pathlib import Path
import sys
import threading

from playwright.sync_api import sync_playwright

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from review_server import build_server


def check(directory):
    root=Path(directory).resolve()
    if (root/'review-state.json').exists():
        raise ValueError('Use a test scan without existing review state')
    server=build_server(root)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}/'
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe',headless=True)
            page=browser.new_page(viewport={'width':1440,'height':1000})
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.add_init_script('''
                window.savedZipBytes = [];
                window.cancelDestination = true;
                window.showSaveFilePicker = async options => {
                    window.suggestedZip = options.suggestedName;
                    if (window.cancelDestination) throw new DOMException('Canceled', 'AbortError');
                    return {
                        name: 'reviewed-conference.zip',
                        createWritable: async () => new WritableStream({
                            write(bytes) { window.savedZipBytes.push(Array.from(bytes.slice(0, 4))); }
                        })
                    };
                };
            ''')
            page.goto(base)
            page.wait_for_function("typeof groups !== 'undefined'")
            page.locator('#slides-tab').click()
            page.locator('#filter').select_option('all')
            page.locator('[data-zoom]').first.click()
            page.wait_for_function("document.querySelector('#lightbox img').complete && document.querySelector('#lightbox img').naturalWidth>1000")
            page.keyboard.press('Escape')
            page.locator('[data-crop]').first.click()
            page.wait_for_function("document.getElementById('cropsource').complete && document.getElementById('cropsource').naturalWidth>1000")
            page.locator('#cx1').fill('2');page.locator('#cy1').fill('2');page.locator('#cx2').fill('98');page.locator('#cy2').fill('98')
            page.locator('#applycrop').click()
            page.wait_for_function('Object.keys(crops).length===1')
            page.wait_for_timeout(300)
            assert (root/'review-state.json').exists()
            page.reload();page.wait_for_function('Object.keys(crops).length===1')
            page.locator('#export').click()
            page.wait_for_function("document.getElementById('status').textContent === 'Export canceled.'")
            assert not list(root.glob('reviewed-*.zip'))
            page.evaluate('window.cancelDestination=false')
            page.locator('#export').click()
            page.wait_for_function("document.getElementById('status').textContent.includes('reviewed-conference.zip')",timeout=90000)
            assert page.evaluate('window.suggestedZip.endsWith("-slides.zip")')
            assert page.evaluate('window.savedZipBytes[0]') == [80, 75, 3, 4]
            download=page.locator('#download').get_attribute('href')
            assert (root/download).is_file()
            second=build_server(root)
            second_thread=threading.Thread(target=second.serve_forever,daemon=True);second_thread.start()
            try:
                page.goto(f'http://127.0.0.1:{second.server_port}/')
                page.wait_for_function('Object.keys(crops).length===1')
            finally:
                second.shutdown();second.server_close();second.slide_source.close();second_thread.join()
            assert not errors,errors
            browser.close()
        print('Browser server checks passed: native image, crop editor, persistent review, canceled and successful Save As; no JavaScript errors.')
    finally:
        server.shutdown();server.server_close();server.slide_source.close();thread.join()


if __name__=='__main__':check(sys.argv[1])
