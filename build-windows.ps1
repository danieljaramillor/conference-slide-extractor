param(
    [string]$Python = "$PSScriptRoot\.venv\Scripts\python.exe",
    [string]$OutputDirectory = "$PSScriptRoot\dist"
)
$ErrorActionPreference = 'Stop'
$originalTcl = $env:TCL_LIBRARY
$originalTk = $env:TK_LIBRARY
Push-Location -LiteralPath $PSScriptRoot
try {
    # Tcl discovery can fail inside a Windows venv even with a full CPython install.
    $pythonBase = & $Python -c 'import sys; print(sys.base_prefix)'
    if ($LASTEXITCODE) { throw 'Could not locate the build Python installation.' }
    $tclRoot = Join-Path $pythonBase 'tcl'
    if (Test-Path -LiteralPath "$tclRoot\tcl8.6\init.tcl") { $env:TCL_LIBRARY = "$tclRoot\tcl8.6" }
    if (Test-Path -LiteralPath "$tclRoot\tk8.6\tk.tcl") { $env:TK_LIBRARY = "$tclRoot\tk8.6" }
    & $Python -c "import tkinter; app=tkinter.Tk(); app.withdraw(); app.destroy()"
    if ($LASTEXITCODE) { throw 'Install Tcl/Tk with the build Python, then rebuild.' }
    $workDirectory = Join-Path "$PSScriptRoot\build" ([guid]::NewGuid().ToString('N'))
    & $Python -m PyInstaller --noconfirm --onefile --noupx --workpath $workDirectory --distpath $OutputDirectory `
        --name ConferenceSlideExtractor-Windows-x64 `
        --add-data 'batch_app.html:.' --add-data 'review_template.html:.' --add-data 'review.js:.' --add-data 'app.css:.' `
        --add-data 'assets/yunet:assets/yunet' `
        --add-data 'assets/practice.zip:assets' --add-data 'docs/server-guide.html:docs' `
        --collect-data rapidocr --collect-data cv2 --collect-binaries onnxruntime `
        --hidden-import rapidocr.inference_engine.onnxruntime `
        --copy-metadata rapidocr --copy-metadata onnxruntime --copy-metadata opencv-python-headless `
        --copy-metadata Pillow --copy-metadata numpy --copy-metadata pyinstaller `
        desktop_app.py
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller exited with code $LASTEXITCODE" }
    $executable = Join-Path $OutputDirectory 'ConferenceSlideExtractor-Windows-x64.exe'
    & $Python "$PSScriptRoot\tests\bundle_contents_check.py" $executable
    if ($LASTEXITCODE) { throw 'Bundle dependency check failed.' }
    $hash = (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  ConferenceSlideExtractor-Windows-x64.exe" | Set-Content -Encoding ascii (Join-Path $OutputDirectory 'SHA256SUMS.txt')
    Get-Item -LiteralPath $executable | Select-Object FullName,Length
}
finally {
    $env:TCL_LIBRARY = $originalTcl
    $env:TK_LIBRARY = $originalTk
    Pop-Location
}
