$TARGET_DIR = "$PSScriptRoot\..\target\coverage"

if (!(Test-Path -Path $TARGET_DIR)) {
    New-Item -ItemType Directory -Path $TARGET_DIR | Out-Null
}

rustc "$PSScriptRoot\coverage.rs" -o "$TARGET_DIR\coverage.exe"
& "$TARGET_DIR\coverage.exe"
Remove-Item "$TARGET_DIR\coverage.exe"