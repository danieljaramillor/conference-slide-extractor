"""End-to-end team package smoke test from ZIP selection through downloaded export."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from playwright.sync_api import sync_playwright


def check(program, python, fixture, screenshot, browser_channel='chrome'):
    # Windows can briefly hold the input ZIP after stopping the frozen process.
    with tempfile.TemporaryDirectory(prefix='extractor-ui-', ignore_cleanup_errors=True) as directory:
        frozen = program.suffix.lower() == '.exe'
        launch = [str(program), '--no-browser'] if frozen else [str(Path(python).resolve()), '-u', str(program)]
        proc = subprocess.Popen([*launch, '--data', directory, '--workers', '2', '--ocr-workers', '2'],
                                cwd=directory, stdout=subprocess.PIPE, text=True, encoding='utf-8')
        try:
            startup = proc.stdout.readline()
            if not startup.startswith('Open http://127.0.0.1:'):
                raise RuntimeError(f'Unexpected application startup: {startup.strip() or "no output"}')
            url = startup.split()[1]
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(channel=browser_channel, headless=True)
                page = browser.new_page(viewport={'width': 1320, 'height': 920}, accept_downloads=True)
                errors, external = [], []
                page.on('pageerror', lambda error: errors.append(str(error)))

                def allow_loopback(route):
                    if route.request.url.startswith(url):
                        route.continue_()
                    else:
                        external.append(route.request.url)
                        route.abort()

                page.route('**/*', allow_loopback)
                page.add_init_script('window.showSaveFilePicker = undefined;')
                page.goto(url)
                guide = page.get_by_role('dialog', name='Extract your first slides')
                guide.wait_for()
                with page.expect_response(lambda response: response.url.endswith('/quick-start/seen') and response.ok):
                    guide.get_by_role('button', name='Use practice ZIP').click()
                assert page.locator('#file-name').inner_text() == 'Practice session.zip'
                page.reload()
                assert not guide.is_visible()
                page.get_by_role('button', name='Quick start').click()
                guide.wait_for()
                page.keyboard.press('Escape')
                guide.wait_for(state='hidden')
                server_guide = page.request.get(url + 'server-guide')
                assert server_guide.ok and 'run-inbox' in server_guide.text()
                page.locator('#archive').set_input_files({'name': 'invalid.zip', 'mimeType': 'application/zip',
                                                         'buffer': b'invalid archive'})
                page.locator('#start').click()
                page.get_by_text('The selected file is not a readable ZIP archive.').first.wait_for()
                page.locator('#archive').set_input_files(fixture)
                page.evaluate("""() => {
                  const input = document.querySelector('#archive');
                  const transfer = new DataTransfer();
                  transfer.items.add(input.files[0]);
                  input.value = '';
                  document.querySelector('#drop-zone').dispatchEvent(
                    new DragEvent('drop', { bubbles: true, cancelable: true, dataTransfer: transfer }));
                }""")
                assert page.locator('#file-name').inner_text() == Path(fixture).name
                page.locator('#start').click()
                page.get_by_role('link', name='Review slides').wait_for(timeout=90_000)
                page.screenshot(path=str(screenshot))
                page.get_by_role('link', name='Review slides').click()
                page.locator('#slides-tab').click()
                assert page.locator('#filter').input_value() == 'included'
                page.wait_for_function("document.querySelectorAll('.card').length === 4", timeout=8000)
                page.locator('#grid [data-zoom]').first.click()
                page.wait_for_function("document.querySelector('#lightbox img').naturalWidth >= 1280")
                page.keyboard.press('Escape')
                assert page.locator('#export').is_hidden()
                page.locator('#continue-step').click()
                page.locator('#export').click()
                page.get_by_role('link', name='Download export ZIP').wait_for(timeout=30_000)
                with page.expect_download() as result:
                    page.get_by_role('link', name='Download export ZIP').click()
                with zipfile.ZipFile(result.value.path()) as downloaded:
                    names = downloaded.namelist()
                    assert len([n for n in names if n.startswith('slides/')]) == 4
                    assert len([n for n in names if n.endswith('.txt')]) == 4
                    assert downloaded.testzip() is None
                page.get_by_role('link', name='Back to batches').click()
                page.get_by_role('link', name='Review slides').wait_for()
                assert not errors, errors
                assert not external, external
                browser.close()
            jobs = [json.loads(file.read_text(encoding='utf-8')) for file in Path(directory).glob('*/batch.json')]
            assert sorted(job['state'] for job in jobs) == ['failed', 'ready']
        finally:
            if frozen and proc.poll() is None:
                # A one-file Windows bundle runs the app in a child bootloader process.
                subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], capture_output=True, check=False)
            else:
                proc.terminate()
            proc.wait(timeout=10)
            proc.stdout.close()
    print('Package GUI passed: invalid ZIP recovery, real OCR, persisted batch status, native preview, export download, local-only browser requests.')


if __name__ == '__main__':
    check(Path(sys.argv[1]).resolve(), sys.argv[2], str(Path(sys.argv[3]).resolve()), Path(sys.argv[4]).resolve())
