# Record the layer-choice test flights written by make_layer_scenarios.py, one after
# another (the renderer keeps the whole map on the GPU, parallel jobs only slow down).
#   powershell -ExecutionPolicy Bypass -File record_layer_tests.ps1
#   ... -Wayback "2014-02-20,2018-01-18"                       # also old Esri imagery
#   ... -Prefix steppe -Bbox "48.161373,32.842047,48.238627,32.957953" -Wayback "2019-04-03"
# A folder that already holds manifest.json is skipped. Output: output\<prefix>_<scenario>[_wayback_<date>]
param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Prefix = "testtopboch",
    [string]$Scenarios = "layer_steps,layer_boundaries,low_hold,sawtooth,between_layers",
    [string]$Wayback = "",
    [string]$Bbox = "48.39950223106055,26.102003186115134,48.44349548338706,26.257266910369705",
    [double]$Speed = 50,
    [int]$Zoom = 18,
    [int]$WaybackZoom = 17,
    [string]$Renderer = "auto",
    # Appended to the Wayback output folders, e.g. -WaybackZoom 18 -NameSuffix "_z18"
    [string]$NameSuffix = ""
)
if (-not (Test-Path $Python)) { $Python = "python" }
$env:PYTHONIOENCODING = "utf-8"
$b = $Bbox -split ","
$dates = $(if ($NameSuffix) { @() } else { @("") }) + @($Wayback -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
foreach ($name in @($Scenarios -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })) {
    $scenario = "scenarios\${Prefix}_$name.json"
    if (-not (Test-Path $scenario)) { Write-Host "no scenario $scenario (run make_layer_scenarios.py first)"; continue }
    $frames = [int]([double]((Get-Content $scenario -Raw | ConvertFrom-Json).duration_s) * 30)
    foreach ($date in $dates) {
        $out = "output\${Prefix}_$name" + $(if ($date) { "_wayback_$date$NameSuffix" } else { "" })
        if (Test-Path "$out\manifest.json") { Write-Host "skip $out"; continue }
        New-Item -ItemType Directory -Force -Path $out | Out-Null
        $extra = if ($date) { @("--map-date", $date, "--zoom", $WaybackZoom) } else { @("--zoom", $Zoom) }
        $started = Get-Date
        & $Python -m simulator.main --mode record --scenario $scenario --speed $Speed @extra `
            --lat_min $b[0] --lon_min $b[1] --lat_max $b[2] --lon_max $b[3] `
            --renderer $Renderer --max-frames $frames --fast --no-display --yes `
            --video-file "$out\video.mp4" --frame-gt-file "$out\video.frames.jsonl" `
            --gt-file "$out\ground_truth.json" --calib-file "$out\calibration.json" `
            --telemetry-file "$out\telemetry.csv" --manifest-file "$out\manifest.json" *> "$out\generation.log"
        Write-Host ("{0,-52} exit={1} {2} min" -f $out, $LASTEXITCODE, [math]::Round(((Get-Date) - $started).TotalMinutes, 1))
    }
}
