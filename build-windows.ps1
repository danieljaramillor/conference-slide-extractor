param(
    [string]$Python = "$PSScriptRoot\.venv\Scripts\python.exe"
)
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    & $Python -m PyInstaller --noconfirm --clean --onefile --noupx `
        --name ConferenceSlideExtractor-Windows-x64 `
        --add-data 'batch_app.html:.' --add-data 'review_template.html:.' --add-data 'review.js:.' `
        --collect-data rapidocr --collect-data cv2 --collect-binaries onnxruntime `
        --hidden-import rapidocr.inference_engine.onnxruntime `
        --copy-metadata rapidocr --copy-metadata onnxruntime --copy-metadata opencv-python-headless `
        --copy-metadata Pillow --copy-metadata numpy --copy-metadata pyinstaller `
        desktop_app.py
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller exited with code $LASTEXITCODE" }
    $executable = Join-Path $PSScriptRoot 'dist\ConferenceSlideExtractor-Windows-x64.exe'
    $hash = (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  ConferenceSlideExtractor-Windows-x64.exe" | Set-Content -Encoding ascii (Join-Path $PSScriptRoot 'dist\SHA256SUMS.txt')
    Get-Item -LiteralPath $executable | Select-Object FullName,Length
}
finally {
    Pop-Location
}
