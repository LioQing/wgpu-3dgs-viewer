$BASE_DIR = $PSScriptRoot
$LCOV_PATH = "$BASE_DIR/lcov.info"

echo "Running coverage..."

cargo llvm-cov clean --workspace

echo "Running 'simple' example"
cargo llvm-cov run --example simple --features="viewer-selection,multi-model" -- -m "$BASE_DIR/model.ply"

echo "Running 'multi-model' example"
cargo llvm-cov run --example multi-model --features="viewer-selection,multi-model" -- -m "$BASE_DIR/model.ply" -m "$BASE_DIR/model.ply"

echo "Running 'selection' example"
cargo llvm-cov run --example selection --features="viewer-selection,multi-model" -- -m "$BASE_DIR/model.ply"

# `--doctests` flag is currently unstable
# echo "Running doctests"
# cargo llvm-cov --no-report --doctests --features="viewer-selection,multi-model"

echo "Running tests"
cargo llvm-cov --no-report nextest --features="viewer-selection,multi-model"

echo "Generating coverage report"
cargo llvm-cov report --lcov --output-path "$LCOV_PATH"

echo "Generating badge"
$total = 0
$covered = 0
Select-String -Path "$LCOV_PATH" -Pattern "DA:" | ForEach-Object {
    if ($_ -match "DA:\d+,(\d+)") {
        $total++
        if ($matches[1] -ne "0") {
            $covered++
        }
    }
}

$badge_percentage = if ($total -eq 0) { 100 } else { [math]::Round(($covered / $total) * 100) }
$badge_color = if ($badge_percentage -ge 80) {
    "brightgreen"
} elseif ($badge_percentage -ge 50) {
    "yellow"
} else {
    "red"
}

"{
    `"schemaVersion`": 1,
    `"label`": `"coverage`",
    `"message`": `"$badge_percentage%`",
    `"color`": `"$badge_color`"
}" | Out-File -FilePath "$BASE_DIR/badge.json" -Encoding ascii

# echo "Cleaning up"
# rm "$BASE_DIR/output.ply"

echo "Done"
